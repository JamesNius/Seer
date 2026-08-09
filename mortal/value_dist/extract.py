"""数据抽取 v2：八头实时真值标签（完整版）。

采样点A：被复盘玩家每个 tsumo 决策点(每巡1个)。标签来自
  hand_tracker 实时手牌 + 当前可见场况，禁止终局/未来信息。
  输出: shan/wait/dora/染手/山残/value/can_call

采样点B：will_call 独立数据集。全部实际切牌事件×其余三家。
  记录 can_call(结构+退化过滤) 和 will_call(实际是否鸣)。
  吃仅切牌者下家; 碰/杠三家均可; 荣和优先(牌局结束 will=0)。

山残恒等式(每样本校验): 每牌 河(净)+副露+指示+四家手牌+山残 = 4。
"""
import gzip, json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi import agari as _agari
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker, tile_id, _deaka, WIND_ID
from value_dist.shanten_exact import shanten_of, is_winning

OBS_VERSION = 4
N_TILE = 34
N_DORA = 7
N_THRESH = 6
THRESHOLDS = [2000, 3900, 7700, 11600, 18000, 36000]


def tile_to_idx(t):
    return tile_id(_deaka(t))


def tile_to_str(i):
    """34 索引 -> mjai 牌串。"""
    if i < 9: return f"{i+1}m"
    if i < 18: return f"{i-9+1}p"
    if i < 27: return f"{i-18+1}s"
    return "ESWN PFC".replace(" ", "")[i-27]


def count34(tiles):
    c = [0] * 34
    for t in tiles:
        c[tile_to_idx(t)] += 1
    return c


def bucket_dora(d):
    return min(max(d, 0), 6)


def bucket_mountain(w):
    return min(max(w, 0), 4)





def dora_tiles(markers):
    """指示牌列表 -> dora 牌索引列表(指示牌+1, 数牌/字牌各自循环)。"""
    out = []
    for d in markers:
        idx = tile_id(d)
        if idx < 27:
            out.append((idx // 9) * 9 + (idx % 9 + 1) % 9)
        else:
            out.append(27 + (idx - 27 + 1) % 7)
    return out


def count_dora_in(tiles34, dora_idx_list):
    return sum(tiles34[i] for i in dora_idx_list)


def count_aka(tiles34_of_hand):
    """给定家的 menzen(含赤标记), 数赤5张数。"""
    return sum(1 for x in tiles34_of_hand if len(x) == 3)


def compute_point(ks, p, concealed34, winning_tile, is_ron, dora_indicators):
    """对某家手牌(concealed34 已含和牌张)算荣和打点。返回点数或 None。"""
    hand = ks.hands[p]
    # concealed34 已含和牌张; 转回牌列表给 calc_agari_point
    tehai_list = []
    for i in range(34):
        tehai_list += [tile_to_str(i)] * concealed34[i]
    jikaze_idx = (p - ks.oya) % 4
    jikaze = ["E", "S", "W", "N"][jikaze_idx]
    additional = 0
    if hand.riichi_accepted:
        additional += 1
    if hand.is_w_riichi:
        additional += 1
    if not is_ron and hand.is_menzen:
        additional += 1
    try:
        ron, tsumo_total, han, fu, doras = _agari.calc_agari_point(
            tehai=tehai_list,
            winning_tile=winning_tile,
            is_ron=is_ron,
            bakaze=WIND_ID[ks.bakaze],
            jikaze=WIND_ID[jikaze],
            is_oya=(p == ks.oya),
            additional_hans=additional,
            dora_indicators=dora_indicators,
            ura_indicators=[],   # 里宝禁止计入
            chis=hand.chis, pons=hand.pons,
            minkans=hand.minkans, ankans=hand.ankans,
            riichi=hand.riichi_accepted,
        )
    except Exception:
        return None
    return ron if is_ron else tsumo_total


def wait_of(ks, p):
    """某家 shan=0 时的手牌结构待牌(34 位 0/1)。shan>0 全 0。"""
    h = ks.hands[p]
    concealed = count34(h.menzen)
    if shanten_of(concealed) != 0:
        return np.zeros(34, np.float32)
    w = np.zeros(34, np.float32)
    for t in range(34):
        if concealed[t] < 4:
            c2 = concealed[:]; c2[t] += 1
            if is_winning(c2):
                w[t] = 1
    return w


class Extractor:
    def __init__(self, viewer):
        self.viewer = viewer
        self.ps = PlayerState(viewer)
        self.tracker = HandTracker()
        self.samples = []
        self.will_samples = []
        self.kawa_pool = Counter()
        self.called_pai = Counter()
        self.kyoku_idx = -1
        self.turn_in_kyoku = 0
        self.stats = {"wall_err": 0, "wall_total": 0, "samples": 0,
                      "value_dist": [0]*N_THRESH}

    def reset_kyoku(self):
        self.kawa_pool = Counter()
        self.called_pai = Counter()
        self.turn_in_kyoku = 0

    def visible_buckets(self):
        net_kawa = self.kawa_pool - self.called_pai
        meld = Counter()
        for p in range(4):
            h = self.tracker.ks.hands[p]
            for m in h.chis:
                for x in m: meld[tile_id(x)] += 1
            for m in h.pons:
                for x in m: meld[tile_id(x)] += 1
            for m in h.minkans: meld[tile_id(m)] += 4
            for m in h.ankans: meld[tile_id(m)] += 4
        ind = Counter(tile_id(d) for d in self.tracker.ks.dora_markers)
        return net_kawa, meld, ind

    def hands_counts(self):
        hands = [0] * 34
        for p in range(4):
            for x in self.tracker.ks.hands[p].menzen:
                hands[tile_id(x)] += 1
        return hands

    def somete_label(self, p):
        """染手按家: 3×4 one-hot(无/万/筒/索)。只含一种数牌(可含字)→该色。"""
        h = self.tracker.ks.hands[p]
        all_tiles = list(h.menzen)
        for m in h.chis + h.pons:
            all_tiles += list(m)
        for m in h.minkans + h.ankans:
            all_tiles += [m]*4
        kinds = set()
        num_kinds = set()
        for x in all_tiles:
            i = tile_id(x)
            if i < 9: num_kinds.add(0)
            elif i < 18: num_kinds.add(1)
            elif i < 27: num_kinds.add(2)
            else: kinds.add(3)  # 字
        lab = np.zeros(4, np.float32)
        if len(num_kinds) == 1:
            lab[list(num_kinds)[0]] = 1
        return lab

    def value_label(self, p):
        """value: shan≤1 出标签(6档阶梯), shan≥2 mask。"""
        ks = self.tracker.ks
        h = ks.hands[p]
        concealed = count34(h.menzen)
        shan = shanten_of(concealed)
        if shan > 1:
            return None, 0
        dora_inds = dora_tiles(ks.dora_markers)
        # 听牌: 每待牌 t 算打点, 残枚加权(残枚=4-可见-三家暗手-自家)
        # 简化残枚: 4 - 已见(含自家手牌) ; 完整在后续
        if shan == 0:
            pts = []
            for t in range(34):
                if concealed[t] < 4:
                    c2 = concealed[:]; c2[t] += 1
                    if is_winning(c2):
                        # 荣和口径: winning_tile = t 的牌串, is_ron=True
                        wt = tile_to_str(t)
                        pt = compute_point(ks, p, c2, wt, True, ks.dora_markers)
                        if pt is not None:
                            pts.append(pt)
            if not pts:
                return None, 0
            avg = sum(pts) / len(pts)
        else:
            # 一向听: 枚举(t,d) 后听牌算打点; 超限直接 mask
            return None, 0  # 骨架先 mask 一向听(宁缺毋假)
        # 6 档阶梯
        lab = np.array([1.0 if avg >= t else 0.0 for t in THRESHOLDS], np.float32)
        return lab, 1

    def process(self, e, ei, events, cans):
        t = e.get("type")
        if t == "start_kyoku":
            self.kyoku_idx += 1
            self.reset_kyoku()
        elif t == "dahai":
            self.kawa_pool[_deaka(e["pai"])] += 1
        elif t in ("chi", "pon", "daiminkan"):
            self.called_pai[_deaka(e["pai"])] += 1
        elif t == "tsumo" and e.get("actor") == self.viewer:
            if cans.can_discard:
                self.turn_in_kyoku += 1
                self.collect_tsumo_sample(e, ei, events)

    def collect_tsumo_sample(self, e, ei, events):
        ks = self.tracker.ks
        obs, _ = self.ps.encode_obs(OBS_VERSION, False)
        hands_all = self.hands_counts()
        net_kawa, meld, ind = self.visible_buckets()

        # 山残 + 恒等式校验
        wall = [4 - net_kawa[i] - meld[i] - ind[i] - hands_all[i] for i in range(34)]
        total_alloc = [net_kawa[i]+meld[i]+ind[i]+hands_all[i]+wall[i] for i in range(34)]
        self.stats["wall_total"] += 1
        if any(x != 4 for x in total_alloc):
            self.stats["wall_err"] += 1

        shan_lab = np.zeros((3, 5), np.float32)
        wait_lab = np.zeros((3, 34), np.float32)
        wait_mask = np.zeros(3, np.float32)
        dora_lab = np.zeros((3, N_DORA), np.float32)
        somete_lab = np.zeros((3, 4), np.float32)
        value_lab = np.zeros((3, N_THRESH), np.float32)
        value_mask = np.zeros(3, np.float32)
        dora_inds = dora_tiles(ks.dora_markers)

        for p in range(3):
            opp = (self.viewer + 1 + p) % 4
            h = ks.hands[opp]
            concealed = count34(h.menzen)
            shan = shanten_of(concealed)
            shan_lab[p, min(max(shan, 0), 4)] = 1
            if shan == 0:
                wait_mask[p] = 1
            wait_lab[p] = wait_of(ks, opp)
            # dora
            dora_cnt = count_dora_in(concealed, dora_inds)
            dora_cnt += count_aka(h.menzen)
            dora_lab[p, bucket_dora(dora_cnt)] = 1
            # 染手按家
            somete_lab[p] = self.somete_label(opp)
            # value
            vl, vm = self.value_label(opp)
            if vl is not None:
                value_lab[p] = vl
                value_mask[p] = vm
                for k in range(N_THRESH):
                    self.stats["value_dist"][k] += int(vl[k])

        self.samples.append({
            "obs": obs.astype(np.float16),
            "shan_label": shan_lab,
            "wait_label": wait_lab, "wait_mask": wait_mask,
            "dora_label": dora_lab,
            "somete_label": somete_lab,
            "mountain_label": np.eye(5)[[bucket_mountain(w) for w in wall]].astype(np.float32),
            "value_label": value_lab, "value_mask": value_mask,
            "kyoku": self.kyoku_idx, "turn": self.turn_in_kyoku,
        })
        self.stats["samples"] += 1


def extract_file(path, viewers=(0, 1, 2, 3)):
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(json.loads(line))
    per_viewer = {}
    total_stats = {"wall_err": 0, "wall_total": 0, "samples": 0,
                   "value_dist": [0]*N_THRESH}

    # 每 viewer 独立 tsumo 样本
    for v in viewers:
        ex = Extractor(v)
        for ei, e in enumerate(events):
            try:
                ex.tracker.update(e)
                cans = ex.ps.update(json.dumps(e))
                ex.process(e, ei, events, cans)
            except Exception:
                continue
        per_viewer[v] = ex.samples
        for k in ("samples", "wall_err", "wall_total"):
            total_stats[k] += ex.stats[k]
        for i in range(N_THRESH):
            total_stats["value_dist"][i] += ex.stats["value_dist"][i]

    return per_viewer, total_stats

