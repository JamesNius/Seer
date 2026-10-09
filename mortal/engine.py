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
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        explore_rate = 0,
        # ── restore #6: 补回被合并丢掉的参数 ──
        # Rust 端 (libriichi/src/agent/mortal.rs) 构造 Bot 时会读这两个属性:
        #   record_mode  : Pass1 录制模式(set_scene 编码但 get_reaction 不 evaluate)
        #   rayon_threads: 编码线程池大小(None=自动核数)
        # mortal_review_server.py 会传 rayon_threads → 不补回会 TypeError。
        rayon_threads = None,
        record_mode = False,
        # ── 复盘兼容: policy(π) 模型 ──────────────────────────────────
        # 复盘(Rust)侧把 q_value 当"分数", 用 prob = softmax(q_value / τ), τ=0.1。
        # policy 模型的原始输出是 π(概率), 直接传过去 prob 会算错。
        # review_policy_tau 不为 None 时, 返回 τ·log(π):
        #     prob = softmax(τ·logπ / τ) = softmax(log π) = π    ← 精确还原
        # 训练/自对弈必须保持原始 π(要按 π 采样) → 默认 None(不转换)。
        review_policy_tau = None,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()   # 
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name
        self.explore_rate = explore_rate
        # ── restore #6 ──
        self.rayon_threads = rayon_threads
        self.record_mode = record_mode
        self.review_policy_tau = review_policy_tau

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
    
        if self.explore_rate > 0:
            is_greedy = torch.full((batch_size,), 1-self.explore_rate, device=self.device).bernoulli().to(torch.bool)
            actions = torch.where(is_greedy, q_out.argmax(-1), Categorical(probs=q_out).sample())
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        if self.review_policy_tau is not None:
            # policy 模型: 转成 τ·log(π), 让复盘侧 softmax(q/τ) 精确还原 π
            q_out = self.review_policy_tau * torch.log(q_out.clamp_min(1e-12))

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

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
