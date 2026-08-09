"""纯 Python 役种判定(用于训练标签, 零 Rust 改动)。

输入: 和了完整手牌(门清含和了牌) + 副露 + 和了牌 + 场况。
输出: 12 役 bool 向量, 顺序 = YAKU_NAMES。

役清单(索引):
  0 立直  1 一发  2 断幺九  3 平和  4 一杯口  5 混一色
  6 清一色  7 七对子  8 对对和  9 三色同顺  10 一气通贯  11 役牌
注: 立直/一发/门清自摸等偶然役由调用方(hand_tracker 的 additional 信息)填入;
    本模块判定牌型役(2..11)。役种口径与 agari.rs::search_yakus 对齐。
"""

YAKU_NAMES = [
    "riichi", "ippatsu", "tanyao", "pinfu", "iipeikou", "honitsu",
    "chinitsu", "chiitoi", "toitoi", "sanshoku", "ittsuu", "yakuhai",
    "chanta", "junchan",
]
N_YAKU = len(YAKU_NAMES)

JIHAI = {"E", "S", "W", "N", "P", "F", "C"}
SANGEN = {"P", "F", "C"}
WINDS = {"E", "S", "W", "N"}


def _deaka(t):
    return t[:2] if len(t) == 3 else t


def _kind_num(t):
    """返回 (kind, num): kind 0/1/2=m/p/s, 3=字牌; num 0-8 (字牌 0-6)。"""
    t = _deaka(t)
    if t in JIHAI:
        return 3, "ESWN PFC".replace(" ", "").index(t)
    suit = "mps".index(t[1])
    return suit, int(t[0]) - 1


def _is_terminal_honor(t):
    k, n = _kind_num(t)
    return k == 3 or n == 0 or n == 8


def _decomp_rest(tiles):
    """拆出全部面子(无雀头), 返回所有可能的面子组合(tuple of mentsu)。"""
    tiles = list(tiles)
    out = []

    def first():
        for i, c in enumerate(tiles):
            if c > 0:
                return i
        return -1

    def rec(mentsu):
        i = first()
        if i == -1:
            out.append(tuple(mentsu))
            return
        if tiles[i] >= 3:
            tiles[i] -= 3
            rec(mentsu + [("koutsu", i)])
            tiles[i] += 3
        k, n = i // 9, i % 9
        if k < 3 and n <= 6 and tiles[i + 1] > 0 and tiles[i + 2] > 0:
            tiles[i] -= 1; tiles[i + 1] -= 1; tiles[i + 2] -= 1
            rec(mentsu + [("shuntsu", i)])
            tiles[i] += 1; tiles[i + 1] += 1; tiles[i + 2] += 1
    rec([])
    return out


def decompose(tiles34):
    """回溯拆分门清手牌(3n+2 计数数组) -> 所有 (pair_id, mentsu_list) 去重组合。"""
    results = []
    seen = set()
    for p in range(34):
        if tiles34[p] >= 2:
            t2 = list(tiles34)
            t2[p] -= 2
            for m in _decomp_rest(t2):
                key = (p, tuple(sorted(m)))
                if key not in seen:
                    seen.add(key)
                    results.append((p, list(m)))
    return results


def _tile_id(t):
    k, n = _kind_num(t)
    return 27 + n if k == 3 else k * 9 + n


def judge_yaku(menzen, chis, pons, minkans, ankans, winning_tile, is_ron,
               bakaze, jikaze, is_menzen, riichi=False, ippatsu=False):
    """返回 N_YAKU 维 0/1 list (顺序=YAKU_NAMES)。menzen=门清手牌(含和了牌)。

    chis/pons = 完整牌列表的列表; minkans/ankans = deaka 牌字符串 list。
    riichi 保留参数以兼容调用方, 但不计入标签(立直可见)。
    """
    flags = [0] * N_YAKU
    idx = {n: i for i, n in enumerate(YAKU_NAMES)}

    # riichi: 作为"该家和了时会带立直"的预测标签(对未来役种的预测)
    flags[idx["riichi"]] = int(riichi)
    flags[idx["ippatsu"]] = int(ippatsu)

    # 门清计数
    cnt = [0] * 34
    for t in menzen:
        cnt[_tile_id(t)] += 1

    # 所有 tile(含副露)
    all_tiles = [_deaka(t) for t in menzen]
    for meld in chis + pons:
        all_tiles += [_deaka(x) for x in meld]
    for k in minkans + ankans:
        all_tiles += [_deaka(k)] * 4

    # 断幺九: 无幺九字牌
    if all(not _is_terminal_honor(t) for t in all_tiles):
        flags[idx["tanyao"]] = 1

    # 混一色/清一色
    kinds = {_kind_num(t)[0] for t in all_tiles}
    has_jihai = 3 in kinds
    num_kinds = len([k for k in kinds if k < 3])
    if num_kinds == 1 and not has_jihai:
        flags[idx["chinitsu"]] = 1
    elif num_kinds == 1 and has_jihai:
        flags[idx["honitsu"]] = 1

    # 七对子: 门清 7 对
    if is_menzen and len(menzen) == 14 and sum(1 for c in cnt if c == 2) == 7:
        flags[idx["chiitoi"]] = 1

    # 拆分(判平和/一杯口/三色/一气/对对/混纯全)
    decomps = decompose(cnt) if sum(cnt) % 3 == 2 else []
    win_id = _tile_id(winning_tile)

    # 对对和: 无吃, 且某拆分全为刻/杠
    if not chis:
        for pair, mentsu in decomps:
            if all(m[0] == "koutsu" for m in mentsu):
                flags[idx["toitoi"]] = 1
                break

    # 副露面子(统一格式)
    open_mentsu = []
    for meld in chis:
        ids = sorted(_tile_id(x) for x in meld)
        open_mentsu.append(("shuntsu", ids[0]))
    for meld in pons:
        open_mentsu.append(("koutsu", _tile_id(meld[0])))
    for k in minkans + ankans:
        open_mentsu.append(("koutsu", _tile_id(k)))

    for pair, mentsu in decomps:
        full = mentsu + open_mentsu

        # 平和: 门清, 4顺子, 雀头非役牌, 两面听
        if is_menzen and not open_mentsu and all(m[0] == "shuntsu" for m in mentsu):
            pair_ok = pair not in (_tile_id("P"), _tile_id("F"), _tile_id("C")) \
                and pair != _tile_id(bakaze) and pair != _tile_id(jikaze)
            if pair_ok:
                for m in mentsu:
                    s = m[1]; num = s % 9
                    if (num <= 6 and s == win_id) or (num >= 1 and s + 2 == win_id):
                        flags[idx["pinfu"]] = 1
        # 一杯口: 门清, 两组相同顺子
        if is_menzen:
            shuntsu_ids = [m[1] for m in mentsu if m[0] == "shuntsu"]
            if len(shuntsu_ids) != len(set(shuntsu_ids)):
                flags[idx["iipeikou"]] = 1
        # 三色同顺
        shuntsu_kinds = {}
        for m in full:
            if m[0] == "shuntsu":
                k, n = m[1] // 9, m[1] % 9
                shuntsu_kinds.setdefault(n, set()).add(k)
        if any(len(v) == 3 for v in shuntsu_kinds.values()):
            flags[idx["sanshoku"]] = 1
        # 一气通贯
        kind_nums = {}
        for m in full:
            if m[0] == "shuntsu":
                k, n = m[1] // 9, m[1] % 9
                kind_nums.setdefault(k, set()).add(n)
        if any({0, 3, 6} <= v for v in kind_nums.values()):
            flags[idx["ittsuu"]] = 1

    # 役牌: 场风/自风/三元 刻或杠
    yakuhai_tiles = {_tile_id("P"), _tile_id("F"), _tile_id("C"),
                     _tile_id(bakaze), _tile_id(jikaze)}
    open_koutsu = {_tile_id(m[0]) for m in pons} | {_tile_id(k) for k in minkans + ankans}
    menzen_koutsu = set()
    if sum(cnt) % 3 == 2:
        for pair, mentsu in decompose(cnt):
            for m in mentsu:
                if m[0] == "koutsu":
                    menzen_koutsu.add(m[1])
    if (open_koutsu | menzen_koutsu) & yakuhai_tiles:
        flags[idx["yakuhai"]] = 1

    # 混全带幺九/纯全带幺九 (非断幺, 非七对子/对对形式的混老/清老)
    # 每组面子+雀头都含幺九: 混全(含字)/纯全(不含字)
    if not flags[idx["tanyao"]] and not flags[idx["chiitoi"]] and not flags[idx["toitoi"]]:
        def mentsu_has_yaochu(m):
            if m[0] == "koutsu":
                k, n = m[1] // 9, m[1] % 9
                return m[1] >= 27 or n in (0, 8)
            else:
                s = m[1]; n = s % 9
                return n in (0, 6)  # 顺子 123 或 789
        def pair_is_yaochu(p):
            return p >= 27 or (p % 9) in (0, 8)
        for pair, mentsu in decomps:
            full = mentsu + open_mentsu
            if not full:
                continue
            if all(mentsu_has_yaochu(m) for m in full) and pair_is_yaochu(pair):
                has_j = any(m[1] >= 27 for m in full) or pair >= 27
                if has_j:
                    flags[idx["chanta"]] = 1
                else:
                    flags[idx["junchan"]] = 1
                break

    return flags
