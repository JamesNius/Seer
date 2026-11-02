# -*- coding: utf-8 -*-
"""HybridEngine —— 按 obs 内的局面特征，在两组权重之间逐样本路由。

两种模式（--gate-mode）：

  early（v3，默认）—— SeerX 只覆盖开局：
      自家巡目 > N                        -> late  (Seer)
      本局已有人立直（含自家）             -> late  (Seer)
      南4 / 南4之后 且 已有人副露或杠      -> late  (Seer)
      其余                                -> early (SeerX)

  wide（v4）—— SeerX 覆盖南4之前的全程：
      南4 及之后:
          前 N 巡 且 无他家立直 且 无副露/杠 -> early (SeerX)
          其余                              -> late  (Seer)
      南4 之前:
          无他家立直                        -> early (SeerX)
          有他家立直（本局剩余永久）         -> late  (Seer)
      ⚠️ wide 只看三家对手的立直，自家立直不触发切换。

为什么所有判据都必须从 obs 里读：
    `react_batch(obs, masks, invisible_obs)` 的 batch 维度是**多局并行**，
    不是一个对局内的多次决策。任何"记住上一手"的全局状态都会被并行局搅乱。
    所以门控必须是 obs 的纯函数。

用到的通道（见 Mortal_v4输入表征_1012通道文档.md，均已用真实 obs 实测校准）：
    ch25       场风，assign 到风牌位：27=东 28=南 29=西 30=北
    ch19-22    局 one-hot，形状 (4,34) —— 必须 argmax(max(axis=1)) 才得到 0..3
               南4  ⟺ ch25==28 且 局索引==3        （注意 ch27 值是 1.0，不是 8/7）
               西入 ⟺ ch25>=29                     （ch27 分不出南和西）
    ch854-856  三家已宣言立直（riichi_declared[1..]，fill 0/1）
    ch857-859  三家立直已成立（riichi_accepted[1..]，fill 0/1）
               —— 两条都读：对手刚打出宣言牌、荣和确认还没走完时只有 declared 为真
    ch869      自家立直已成立（riichi_accepted[0]）
    ch878      自家立直已宣言（riichi_declared[0]，覆盖"已宣言未成立"的窗口）
    ch751-830  四家副露，每家 4 槽 x 5 通道（前 4 = 该组牌，第 5 = 含赤）
    ch831-834  四家暗杠
               见 obs_repr.rs:303-329；任意非零 = 已有副露/杠
    ch59-130   自家河·最近 18 巡，每巡 4 通道（索引 1 = 舍牌 assign）
               -> 有任意非零 = 该巡已有动作；数占用槽数得到已完成巡数
               （实测：与 draws//4+1 偏差均值 0.28、±1 内 100%；ch35-58 那块会饱和，不可用）
"""
from typing import List, Optional, Tuple

import numpy as np


# ── obs 通道常量 ─────────────────────────────────────────────────────────
CH_KYOKU_PROG = 27
CH_FIELD_WIND = 25                  # 场风 assign 到风牌位: 27=东 28=南 29=西 30=北
CH_KYOKU_START, CH_KYOKU_END = 19, 23   # 局 one-hot，形状 (4,34)，必须按行取 argmax
HONOR_EAST, HONOR_SOUTH, HONOR_WEST, HONOR_NORTH = 27, 28, 29, 30
CH_RIICHI_OTHERS = (854, 855, 856, 857, 858, 859)   # 三家: 已宣言 + 已成立
CH_RIICHI_SELF_ACCEPTED = 869                        # 自家: 已成立 (riichi_accepted[0])
CH_RIICHI_SELF_DECLARED = 878                        # 自家: 已宣言 (riichi_declared[0])
# 副露 + 暗杠（见 libriichi/src/state/obs_repr.rs:303-329）
#   ch751-830 = 四家副露，每家 4 槽 x 5 通道（前 4 = 该组的牌，第 5 = 含赤）
#   ch831-834 = 四家暗杠
CH_FURO_START, CH_FURO_END = 751, 831
CH_ANKAN_START, CH_ANKAN_END = 831, 835

# 自家河：用「最近 18 巡」块 (ch59-130，18 槽 x 4 通道) 数占用槽数。
# ⚠️ 不要用 ch35-58「前 6 巡」块 —— 实测它会在 6 饱和（49% 样本卡在 6），
#    且槽 1 占用率仅 0.274 而槽 2 是 0.925，明显不是按时间顺序排列。
#    18 槽块与独立参考 (draws//4+1) 的偏差均值 0.28、±1 内 100%；6 槽块是 2.82 / 52%。
SELF_KAWA_RECENT_START = 59
SELF_KAWA_RECENT_END = 131
SELF_KAWA_ITEM = 4
N_SLOTS_RECENT = (SELF_KAWA_RECENT_END - SELF_KAWA_RECENT_START) // SELF_KAWA_ITEM  # 18


def field_wind(obs: np.ndarray) -> int:
    """当前场风的风牌索引：27=东 28=南 29=西 30=北；读不到返回 -1。"""
    v = obs[CH_FIELD_WIND]
    if float(np.max(v)) <= 0.0:
        return -1
    return int(np.argmax(v))


def kyoku_index(obs: np.ndarray) -> int:
    """当前场内的第几局，0 起（0=东1 / 南1，3=东4 / 南4）；读不到返回 -1。

    ⚠️ ch19-22 是 (4,34) 的"局 one-hot"，直接 np.argmax 会拍平成 0/34/68/102，
       必须先按行取 max 再 argmax，才得到 0..3。
    """
    blk = obs[CH_KYOKU_START:CH_KYOKU_END]
    if blk.size == 0:
        return -1
    per_row = np.max(blk, axis=1)
    if float(per_row.max()) <= 0.0:
        return -1
    return int(np.argmax(per_row))


def is_south4(obs: np.ndarray) -> bool:
    """南 4 局。

    实测对应关系（737 样本，4 半庄）：
        ch27 = (场风东?0:1)*4 + 局索引) / 7
        东1..东4 -> 0, 1/7, 2/7, 3/7      南1..南4 -> 4/7, 5/7, 6/7, **1.0**
    ⇒ 南4 = 1.0（7/7），不是 8/7。
    """
    return field_wind(obs) == HONOR_SOUTH and kyoku_index(obs) == 3


def is_after_south4(obs: np.ndarray) -> bool:
    """南 4 之后（西入 / 北入）。ch27 分不出南和西（都用 (1*4+局)/7），必须看 ch25 场风。"""
    return field_wind(obs) >= HONOR_WEST


def is_final_stage(obs: np.ndarray) -> bool:
    """南 4 及之后，一律切 Seer。"""
    return is_south4(obs) or is_after_south4(obs)


def has_opponent_riichi(obs: np.ndarray) -> bool:
    """只看三家对手是否已有人立直（不含自家）。wide 模式用这个。

    ch854-856 = riichi_declared[1..]（已宣言）、ch857-859 = riichi_accepted[1..]（已成立）。
    两条都读 —— 对手刚打出立直宣言牌、荣和确认还没走完的那一刻只有 declared 为真，
    而那恰恰是最需要防守的一手。
    """
    return float(np.max(obs[list(CH_RIICHI_OTHERS)])) > 0.5


def has_any_riichi(obs: np.ndarray) -> bool:
    """本局是否已有人立直（含自家）。early 模式用这个。

    自家：ch869 = accepted[0]、ch878 = declared[0]（已宣言未成立）。
    """
    if float(np.max(obs[CH_RIICHI_SELF_ACCEPTED])) > 0.5:
        return True
    if float(np.max(obs[CH_RIICHI_SELF_DECLARED])) > 0.5:
        return True
    return has_opponent_riichi(obs)


def has_any_riichi_v21(obs: np.ndarray) -> bool:
    """v2.1 口径：自家只认 riichi_accepted[0]（ch869），**不含** ch878。

    用于 --base v21：与当年跑出 +1.472 的那版逐位一致，方便单变量对比。
    """
    if float(np.max(obs[CH_RIICHI_SELF_ACCEPTED])) > 0.5:
        return True
    return has_opponent_riichi(obs)


def turn_index(obs: np.ndarray) -> int:
    """自家当前巡目（1 起）。

    口径：自家河「最近 18 巡」的已占用槽数 + 1。
    每个自己的回合都以"自己打出一张"结束（含副露后、暗杠后），所以槽数 == 已完成的巡数。
    数占用槽数与槽的排列顺序无关，因此不受该块是顺序/倒序/环形的影响。
    """
    n = 0
    for i in range(SELF_KAWA_RECENT_START, SELF_KAWA_RECENT_END, SELF_KAWA_ITEM):
        if float(np.max(obs[i:i + SELF_KAWA_ITEM])) > 0.0:
            n += 1
    return n + 1


def has_call_or_kan(obs: np.ndarray) -> bool:
    """是否已经有人吃/碰/明杠（副露）或暗杠。

    副露 ch751-830：每家 4 槽，每槽 5 通道，只要有任意非零即存在副露。
    暗杠 ch831-834：单独四通道，同一并计入（杠本身就是"局面已经展开"的信号）。
    """
    return float(np.max(obs[CH_FURO_START:CH_ANKAN_END])) > 0.0


GATE_MODES = ('early', 'wide')
BASE_VARIANTS = ('v3', 'v21')


def gate_decision(obs: np.ndarray, gate_turn: int = 6, mode: str = 'early',
                  late_from: int = 0, base: str = 'v3'):
    """返回 (是否用 early/SeerX, 原因)。

    原因 ∈ {'ok', 'ok_late', 'riichi', 'turn_over', 'final_meld', 'final'}

    base 决定"南4 及之后"和"自家立直"怎么算：
        'v3'  —— 南4+ 需"无副露/杠"才用 SeerX；自家立直含 ch878（已宣言未成立）
        'v21' —— 南4+ **一律 Seer**；自家立直只认 ch869（与 +1.472 那一版逐位一致）

    late_from > 0 时再开一个"深巡"窗口：**仅在南4 之前**，巡目 >= late_from 且平静
    也用 SeerX。南4/南4以后不吃这条（终盘仍只看前 gate_turn 巡）。
    """
    if base not in BASE_VARIANTS:
        raise ValueError(f'unknown base: {base!r} (expect {BASE_VARIANTS})')

    t = turn_index(obs)
    final = is_final_stage(obs)

    if mode == 'wide':
        riichi = has_opponent_riichi(obs)          # 只看三家对手
        if not final:
            return (not riichi), ('riichi' if riichi else 'ok')
        if riichi:
            return False, 'riichi'
        if has_call_or_kan(obs):
            return False, 'final_meld'
        return (t <= gate_turn), ('ok' if t <= gate_turn else 'turn_over')

    if mode != 'early':
        raise ValueError(f'unknown gate mode: {mode!r} (expect {GATE_MODES})')

    riichi = has_any_riichi(obs) if base == 'v3' else has_any_riichi_v21(obs)
    if riichi:
        return False, 'riichi'
    if final:
        if base == 'v3':
            if has_call_or_kan(obs):
                return False, 'final_meld'
        else:
            return False, 'final'                  # v2.1: 南4 一律 Seer
    if t <= gate_turn:
        return True, 'ok'
    if late_from > 0 and t >= late_from and not final:
        return True, 'ok_late'
    return False, 'turn_over'


def use_early(obs: np.ndarray, gate_turn: int = 6, mode: str = 'early',
              late_from: int = 0, base: str = 'v3') -> bool:
    """True = 用 early(SeerX)；False = 用 late(Seer)。"""
    return gate_decision(obs, gate_turn, mode, late_from, base)[0]


def explain(obs: np.ndarray, gate_turn: int = 6, mode: str = 'early',
            late_from: int = 0) -> str:
    """给单条 obs 一句人话解释（调试用）。"""
    e, why = gate_decision(obs, gate_turn, mode, late_from, base)
    self_riichi = (float(np.max(obs[CH_RIICHI_SELF_DECLARED])) > 0.5
                   or float(np.max(obs[CH_RIICHI_SELF_ACCEPTED])) > 0.5)
    return (f'mode={mode} base={base} late_from={late_from} wind={field_wind(obs)} '
            f'kyoku={kyoku_index(obs)} final={is_final_stage(obs)} '
            f'opp_riichi={has_opponent_riichi(obs)} self_riichi={self_riichi} '
            f'call={has_call_or_kan(obs)} turn={turn_index(obs)} '
            f'-> {"early" if e else "late"} ({why})')


class HybridEngine:
    """把两个 MortalEngine 包成一个：对外接口与 MortalEngine 完全一致。

    - react_batch 是唯一的决策入口，逐样本路由。
    - 其余属性（engine_type / record_mode / rayon_threads / ...）一律转发给 late，
      因为 PyO3 侧会直接读引擎属性，不转发会报
      "'function' object cannot be converted to 'PyString'"。
    """

    def __init__(self, early, late, gate_turn: int = 6, name: Optional[str] = None,
                 mode: str = 'early', late_from: int = 0, base: str = 'v3'):
        if mode not in GATE_MODES:
            raise ValueError(f'unknown gate mode: {mode!r} (expect {GATE_MODES})')
        if base not in BASE_VARIANTS:
            raise ValueError(f'unknown base: {base!r} (expect {BASE_VARIANTS})')
        # 用 object.__setattr__ 写，避免 __getattr__ 在初始化期间递归
        object.__setattr__(self, '_early', early)
        object.__setattr__(self, '_late', late)
        object.__setattr__(self, 'gate_turn', int(gate_turn))
        object.__setattr__(self, 'gate_mode', mode)
        object.__setattr__(self, 'late_from', int(late_from))
        object.__setattr__(self, 'gate_base', base)
        object.__setattr__(self, 'engine_type', getattr(late, 'engine_type', 'mortal'))
        tag = mode
        if late_from:
            tag += f'+late{late_from}'
        if base != 'v3':
            tag += f'+{base}'
        object.__setattr__(self, 'name',
                           name or f'Hybrid[{early.name}>{late.name}|{tag}]')
        object.__setattr__(self, 'oracle_log', [])
        object.__setattr__(self, '_stat', {'early': 0, 'late': 0,
                                           'why_turn': 0, 'why_late': 0,
                                           'why_riichi': 0, 'why_final': 0,
                                           'why_final_meld': 0, 'why_turn_over': 0,
                                           'n_final': 0, 'n_final_seerx': 0})

    # ── 属性转发 ──────────────────────────────────────────────────────
    def __getattr__(self, item):
        if item in ('_late', '_early'):
            raise AttributeError(item)
        return getattr(object.__getattribute__(self, '_late'), item)

    # ── 决策 ──────────────────────────────────────────────────────────
    def react_batch(self, obs, masks, invisible_obs=None):
        n = len(obs)
        sel_e = [False] * n
        st = self._stat
        for i in range(n):
            o = np.asarray(obs[i])
            early_sel, why = gate_decision(o, self.gate_turn, self.gate_mode,
                                           self.late_from, self.gate_base)
            if is_final_stage(o):
                st['n_final'] += 1
            if early_sel:
                st['why_late' if why == 'ok_late' else 'why_turn'] += 1
                sel_e[i] = True
                if is_final_stage(o):
                    st['n_final_seerx'] += 1
            elif why == 'riichi':
                st['why_riichi'] += 1
            elif why == 'final_meld':
                st['why_final_meld'] += 1
            elif why == 'final':
                st['why_final'] += 1
            else:                                   # 'turn_over'
                st['why_turn_over'] += 1

        idx_e = [i for i in range(n) if sel_e[i]]
        idx_l = [i for i in range(n) if not sel_e[i]]
        st['early'] += len(idx_e)
        st['late'] += len(idx_l)

        def _sub(engine, idx):
            if not idx:
                return None
            sub_obs = [obs[i] for i in idx]
            sub_masks = [masks[i] for i in idx]
            sub_inv = None if invisible_obs is None else [invisible_obs[i] for i in idx]
            return engine.react_batch(sub_obs, sub_masks, sub_inv)

        out_e = _sub(self._early, idx_e)
        out_l = _sub(self._late, idx_l)

        actions: List = [None] * n
        qs: List = [None] * n
        ms: List = [None] * n
        gr: List = [None] * n
        for src, idx in ((out_e, idx_e), (out_l, idx_l)):
            if src is None:
                continue
            a, q, m, g = src
            for k, i in enumerate(idx):
                actions[i] = a[k]
                qs[i] = q[k]
                ms[i] = m[k]
                gr[i] = g[k]
        return actions, qs, ms, gr

    # ── 诊断 ──────────────────────────────────────────────────────────
    def stats(self):
        d = dict(self._stat)
        total = d['early'] + d['late']
        d['total'] = total
        d['early_pct'] = (100.0 * d['early'] / total) if total else 0.0
        return d

    def reset_stats(self):
        for k in self._stat:
            self._stat[k] = 0
