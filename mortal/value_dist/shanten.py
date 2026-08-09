"""纯 Python 真向听计算（DP 实现），对齐 libriichi calc_all 语义。

入口：calc_shanten(concealed[34], len_div3) -> int
  - concealed: 34 计数数组（暗手，不含任何副露牌；赤已归一为普通牌）
  - len_div3: 4 - 副露组数（含暗杠/明杠/碰/吃，每副露 -1）
  - 返回真向听；调用方 .max(0) 对齐 PlayerState
  - 七对子/国士仅 len_div3==4（门清）参与取 min

口径验证：PlayerState.shanten() = calc_all(tehai暗手, len_div3).max(0)，
暗杠也移出暗手 + len_div3-1。本 DP 与该口径逐一手牌对拍，闸门 100%。

DP 结构：每数牌花色独立枚举面子/搭子/雀头/弃子，字牌单独，
背包合并三花色+字牌，全局约束雀头≤1、面子+搭子≤len_div3。
公式：shanten = 2*len_div3 - 2*面子 - 生效搭子 - 雀头。
"""

# 花色每格: 该花色 9 张(0-8) 计数
SUITS = [(0, 9), (9, 18), (18, 27)]   # m, p, s 在 34 数组的起止


def _chiitoitsu(hand):
    pairs = 0
    kinds = 0
    for c in hand:
        if c > 0:
            kinds += 1
            if c >= 2:
                pairs += 1
    return 7 - pairs + max(0, 7 - kinds) - 1


def _kokushi(hand):
    kinds = 0
    has_pair = 0
    for i in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33):
        if hand[i] > 0:
            kinds += 1
            if hand[i] >= 2:
                has_pair = 1
    return 14 - kinds - has_pair - 1


def _suit_dp(cells):
    """单花色 DP。cells: 9 格计数。
    返回 dict[(m, t, p)] = True, m=面子数 t=搭子数 p=是否占雀头。
    枚举该花色所有 面子/搭子(对子/边/坎/两面)/雀头/弃子 分解。
    """
    reachable = {(0, 0, 0)}
    # 递归生成该花色所有"消费"方式
    # 用 (面子, 搭子, 雀头) 三元组 + 剩余 cells
    def rec(i, m, t, p, rem):
        # i 从 0..9 扫, rem = 剩余计数副本
        if i >= 9:
            reachable.add((m, t, p))
            return
        if rem[i] == 0:
            rec(i + 1, m, t, p, rem)
            return
        cnt = rem[i]
        # 弃掉这张
        rem2 = rem[:]; rem2[i] -= 1
        rec(i, m, t, p, rem2)
        # 刻子(3张)
        if cnt >= 3:
            rem3 = rem[:]; rem3[i] -= 3
            rec(i, m + 1, t, p, rem3)
        # 雀头(2张, 全手至多1, 用 p 标记)
        if cnt >= 2 and p == 0:
            rem4 = rem[:]; rem4[i] -= 2
            rec(i, m, t, 1, rem4)
        # 对子当搭子(2张)
        if cnt >= 2:
            rem5 = rem[:]; rem5[i] -= 2
            rec(i, m, t + 1, p, rem5)
        # 顺子: i, i+1, i+2 都有
        if i <= 6 and rem[i + 1] > 0 and rem[i + 2] > 0:
            rem6 = rem[:]; rem6[i] -= 1; rem6[i + 1] -= 1; rem6[i + 2] -= 1
            rec(i, m + 1, t, p, rem6)
        # 两面/边张搭子: i, i+1
        if i <= 7 and rem[i + 1] > 0:
            rem7 = rem[:]; rem7[i] -= 1; rem7[i + 1] -= 1
            rec(i, m, t + 1, p, rem7)
        # 坎张搭子: i, i+2
        if i <= 6 and rem[i + 2] > 0:
            rem8 = rem[:]; rem8[i] -= 1; rem8[i + 2] -= 1
            rec(i, m, t + 1, p, rem8)
    rec(0, 0, 0, 0, cells)
    return reachable


def _honor_dp(hand):
    """字牌 DP(27-33): 只有刻子/对子/弃子。返回 (m, t, p) 可达集。"""
    reachable = {(0, 0, 0)}
    for i in range(27, 34):
        c = hand[i]
        new = set()
        for m, t, p in reachable:
            # 弃
            new.add((m, t, p))
            # 刻子
            if c >= 3:
                new.add((m + 1, t, p))
            # 对子(雀头 或 搭子)
            if c >= 2:
                if p == 0:
                    new.add((m, t, 1))
                new.add((m, t + 1, p))
        reachable = new
    return reachable


def _calc_standard(concealed, len_div3):
    """标准形向听。返回最小 shanten。"""
    # 各花色 DP
    suit_sets = []
    for a, b in SUITS:
        cells = concealed[a:b]
        suit_sets.append(_suit_dp(cells))
    honor = _honor_dp(concealed)
    # 背包合并: 4 组 → (总面子, 总搭子, 总雀头)
    best = None  # 找最大(面子, 搭子) 组合, 雀头优先
    # 先三花色合并
    cur = {(0, 0, 0)}
    for s in suit_sets:
        nxt = set()
        for m1, t1, p1 in cur:
            for m2, t2, p2 in s:
                nxt.add((m1 + m2, t1 + t2, p1 + p2))
        cur = nxt
    # 再合并字牌
    final = set()
    for m1, t1, p1 in cur:
        for m2, t2, p2 in honor:
            final.add((m1 + m2, t1 + t2, p1 + p2))
    # 全局约束: 面子+搭子 ≤ len_div3 (超额搭子丢弃), 雀头 ≤ 1
    # 公式: shanten = 2*len_div3 - 2*面子 - 生效搭子 - 雀头
    best_shanten = 8
    for m, t, p in final:
        if p > 1:
            continue
        eff_t = min(t, len_div3 - m)
        if eff_t < 0:
            continue  # 面子已超 len_div3, 不可能(实际不会)
        sh = 2 * len_div3 - 2 * m - eff_t - p
        if sh < best_shanten:
            best_shanten = sh
    return best_shanten


def calc_shanten(concealed, len_div3):
    """统一真向听入口。concealed[34] 计数, len_div3=4-副露组数。"""
    if len_div3 == 4:
        chiitoi = _chiitoitsu(concealed)
        if chiitoi < 0:
            return chiitoi
        kokushi = _kokushi(concealed)
        if kokushi < 3:
            return kokushi
        standard = _calc_standard(concealed, len_div3)
        return min(standard, chiitoi, kokushi)
    return _calc_standard(concealed, len_div3)
