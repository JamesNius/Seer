import json
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *

class MortalEngine:
    def __init__(
            self,
            brain,
            dqn,
            is_oracle,
            version,
            oracle_net = None,
            device = None,
            stochastic_latent = False,
            enable_amp = False,
            enable_quick_eval = True,
            enable_rule_based_agari_guard = False,
            name = 'NoName',
            boltzmann_epsilon = 0,
            boltzmann_temp = 1,
            top_p = 1,
            rayon_threads = None,
            record_mode = False,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        # ── oracle 侧信道：None 时零开销零行为变化 ──
        self.oracle_net = oracle_net.to(self.device).eval() if oracle_net is not None else None
        self.oracle_log = []

        # ── 编码 rayon 线程池大小（None=默认核数）；Rust 端 Bot 构造时读取 ──
        self.rayon_threads = rayon_threads
        # ── Pass 1 录制模式（record_mode=True 时 Rust 端 set_scene 编码但
        #    get_reaction 不 evaluate，省掉每决策点同步等待）──
        self.record_mode = record_mode

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        batch_size = obs.shape[0]

        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4:
                phi = self.brain(obs)
                q_out = self.dqn(phi, masks)

        # ── oracle：复用本次 phi，追加到侧信道日志（不进返回值）──
        if self.oracle_net is not None:
            feat = latent if self.version == 1 else phi
            self.oracle_log.extend(pack_oracle(self.oracle_net(feat)))

        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

def pack_oracle(logits):
    """oracle五头 → per-step JSON。返回 list，长度=batch_size，每元素为3家字典。"""
    o_cnt, o_aka, o_shan, o_wait, o_furi = logits
    B = o_cnt.shape[0]
    dev = o_cnt.device
    cnt_p = o_cnt.reshape(B, 3, 34, 5).softmax(-1)
    ar5 = torch.arange(5, dtype=cnt_p.dtype, device=dev)
    ar7 = torch.arange(7, dtype=cnt_p.dtype, device=dev)
    exp_cnt = (cnt_p * ar5).sum(-1)          # (B,3,34) 每牌期望张数
    shan_p = o_shan.softmax(-1)              # (B,3,7)
    exp_shan = (shan_p * ar7).sum(-1)        # (B,3)   期望向听
    wait_p = o_wait.reshape(B, 3, 34).sigmoid()
    aka_p = o_aka.reshape(B, 3, 3).sigmoid()
    furi_p = o_furi.reshape(B, 3).sigmoid()
    out = []
    for b in range(B):
        players = []
        for k in range(3):   # 0下家 1对家 2上家
            players.append({
                'exp_cnt': [round(x, 3) for x in exp_cnt[b, k].tolist()],
                'shan': round(float(exp_shan[b, k]), 2),
                'wait': [round(x, 3) for x in wait_p[b, k].tolist()],
                'aka': [round(x, 3) for x in aka_p[b, k].tolist()],
                'furi': round(float(furi_p[b, k]), 3),
            })
        out.append(players)
    return out

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass