# -*- coding: utf-8 -*-
"""
value_dist 向听计算标准件（最终定论版）
=================================================
结论（暴力枚举裁决，勿再自研向听算法）：
  - 唯一可信源 = PyPI mahjong 库（v2.0.0，MIT，MahjongRepository）
    其标准形是经典 sphair 系精确算法，且原生支持副露手
    （按 张数//3 自动推 len_div3，支持 1/2/4/5/7/8/10/11/13/14 张）
  - 已验证：对子重型争议手牌（lib≤2 区域）暴力枚举全部通过
  - libriichi calc_all 与真向听存在口径差异（13 张手牌系统性 +1、
    七对/国士不参与），禁止再以 PlayerState.shanten() 为对齐目标
用法：
  from mahjong.shanten import Shanten
  _lib = Shanten()
  shanten_of(concealed_counts34)          # 全形态真向听（标准/七对/国士 min）
  shanten_regular(concealed_counts34)     # 仅标准形（调试用）
注意：
  - 传入的是【暗手】34 计数，副露牌不进数组；张数必须 % 3 != 0
  - 赤牌进函数前归一为普通牌（0m/0p/0s -> 5m/5p/5s）
  - 假设鸣牌推演：鸣后暗手(10/11 张)直接传，len_div3 自动正确
  - 七对/国士仅门清(≥13 张)自动参与 min；副露手自动只用标准形
    ——这正是 can_call 退化过滤需要的语义，无需额外判断
  - 性能 ~27μs/次，千万次级调用配多进程即可
"""
from mahjong.shanten import Shanten

_lib = Shanten()


def shanten_of(concealed_counts34):
    """全形态真向听。-1=和了 0=听牌。concealed_counts34: 暗手 34 计数（%3!=0）"""
    return _lib.calculate_shanten(concealed_counts34)


def shanten_regular(concealed_counts34):
    """仅标准形向听（对照/调试用）"""
    return _lib.calculate_shanten_for_regular_hand(concealed_counts34)


# ============================================================
# 暴力校验器（独立真值，与任何向听算法无关）
# 用于抽检：is_winning / is_tenpai / path_exists_k
# ============================================================
from functools import lru_cache


def is_winning(counts14):
    """精确和牌判定（14 张）: 标准形(4面子+雀头) 或 七对子 或 国士。"""
    c = tuple(counts14)

    # 七对子: 7 组对子
    if all(x in (0, 2) for x in c) and sum(1 for x in c if x == 2) == 7:
        return True

    # 国士: 13 种幺九/字牌各1 + 任意1种2张
    KOKUSHI = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    if all(c[i] >= 1 for i in KOKUSHI):
        extra = sum(c[i] for i in KOKUSHI) - 13
        if extra == 1:
            return True

    @lru_cache(maxsize=None)
    def decompose(state):
        i = next((k for k in range(34) if state[k] > 0), None)
        if i is None:
            return True
        s = list(state)
        if s[i] >= 3:
            s[i] -= 3
            if decompose(tuple(s)):
                return True
            s[i] += 3
        if i < 27 and i % 9 <= 6 and s[i + 1] > 0 and s[i + 2] > 0:
            s[i] -= 1; s[i + 1] -= 1; s[i + 2] -= 1
            if decompose(tuple(s)):
                return True
            s[i] += 1; s[i + 1] += 1; s[i + 2] += 1
        return False

    for j in range(34):
        if c[j] >= 2:
            s = list(c); s[j] -= 2
            if decompose(tuple(s)):
                return True
    return False


def is_tenpai(counts13):
    """13 张听牌判定：存在一张牌使手和了"""
    for w in range(34):
        if counts13[w] < 4:
            c = list(counts13); c[w] += 1
            if is_winning(c):
                return True
    return False


def path_exists_k(counts13, k):
    """是否存在 k 步（摸切）内听牌的路径，k∈{1,2}。用于抽检 lib 低向听区"""
    if k == 1:
        for t in range(34):
            if counts13[t] >= 4:
                continue
            c = list(counts13); c[t] += 1
            for d in range(34):
                if c[d] == 0:
                    continue
                c2 = list(c); c2[d] -= 1
                if is_tenpai(c2):
                    return True
        return False
    for t1 in range(34):
        if counts13[t1] >= 4:
            continue
        c = list(counts13); c[t1] += 1
        for d1 in range(34):
            if c[d1] == 0:
                continue
            c1 = list(c); c1[d1] -= 1
            for t2 in range(34):
                if c1[t2] >= 4:
                    continue
                c2 = list(c1); c2[t2] += 1
                for d2 in range(34):
                    if c2[d2] == 0:
                        continue
                    c3 = list(c2); c3[d2] -= 1
                    if is_tenpai(c3):
                        return True
    return False
