"""逐事件重建 4 家完整手牌与场况,供 hora 时用 libriichi.agari 精确算番。

设计要点:
- mjai 的 hora 事件不含和了手牌,必须从 start_kyoku 的 tehais 起随事件演化重建。
- 只需重建"足以算番"的信息: 门清手牌、各副露(完整牌)、立直/一发/岭上等偶然役、
  dora/ura 指示牌、场/自风、是否庄家。
- 打点口径: 荣和名义打点(point.ron); 自摸用 point.tsumo_total 折算成荣和等价。
  均不含本场/供托。立直和了计入 ura_markers 翻开的里宝。

仅依赖 libriichi 的 tile 字符串约定(1m..9m,5mr,E/S/W/N/P/F/C)。
"""

from dataclasses import dataclass, field

from libriichi import agari as _agari

from value_dist.yaku import judge_yaku

JIHAI = {"E", "S", "W", "N", "P", "F", "C"}
# 场风/自风 mjai 字牌 -> libriichi 期望的 u8 id (E=27,S=28,W=29,N=30)
WIND_ID = {"E": 27, "S": 28, "W": 29, "N": 30}


def _deaka(t: str) -> str:
    """5mr -> 5m"""
    return t[:2] if len(t) == 3 else t


def tile_id(t: str) -> int:
    """mjai 字符串 -> tile_id 0-33 (m0-8 p9-17 s18-26 字27-33)。"""
    t = _deaka(t)
    if t in JIHAI:
        return 27 + "ESWN PFC".replace(" ", "").index(t)
    return "mps".index(t[1]) * 9 + int(t[0]) - 1


@dataclass
class PlayerHand:
    menzen: list = field(default_factory=list)   # 门清手牌(含赤标记), 含摸进未打出的牌
    chis: list = field(default_factory=list)      # 每个 chi 的完整牌列表(含被吃牌)
    pons: list = field(default_factory=list)      # 每个 pon 的完整牌列表
    minkans: list = field(default_factory=list)   # 大明杠/加杠 的牌(deaka)
    ankans: list = field(default_factory=list)    # 暗杠 的牌(deaka)
    riichi_declared: bool = False
    riichi_accepted: bool = False
    is_w_riichi: bool = False       # 双立直
    ippatsu: bool = False           # 一发窗口(立直后到自己下次打牌前, 无人鸣牌)
    at_rinshan: bool = False        # 当前摸牌是岭上牌
    # 鸣牌行为记录(训练标签): 本局实际鸣过的牌 tile_id(0-33) 计数
    called_ponkan: list = field(default_factory=lambda: [0] * 34)  # 碰/杠
    called_chi: list = field(default_factory=lambda: [0] * 34)     # 吃

    @property
    def is_menzen(self) -> bool:
        return not self.chis and not self.pons and not self.minkans


@dataclass
class KyokuState:
    bakaze: str = "E"
    kyoku: int = 1
    honba: int = 0
    oya: int = 0
    dora_markers: list = field(default_factory=list)
    hands: list = field(default_factory=lambda: [PlayerHand() for _ in range(4)])
    tiles_left: int = 70
    last_discarder: int = -1
    chankan_chance: bool = False    # 加杠瞬间可能被枪杠
    # 记录每家本局是否在第一巡就立直(双立直判定用)
    first_turn_discard: list = field(default_factory=lambda: [False] * 4)
    any_call_made: bool = False     # 本局是否已有人鸣牌(破坏双立直/一发)


class HandTracker:
    """驱动一局内的事件,重建算番所需信息。"""

    def __init__(self):
        self.ks = None

    def update(self, e: dict):
        t = e.get("type")
        fn = getattr(self, f"_on_{t}", None)
        if fn is not None:
            fn(e)

    # --- kyoku lifecycle ---
    def _on_start_kyoku(self, e):
        ks = KyokuState(
            bakaze=e["bakaze"],
            kyoku=e.get("kyoku", 1),
            honba=e.get("honba", 0),
            oya=e["oya"],
            dora_markers=[e["dora_marker"]],
        )
        ks.hands = [PlayerHand(menzen=sorted(tehai)) for tehai in e["tehais"]]
        self.ks = ks

    def _on_dora(self, e):
        self.ks.dora_markers.append(e["dora_marker"])

    # --- draws / discards ---
    def _on_tsumo(self, e):
        ks = self.ks
        actor = e["actor"]
        pai = e["pai"]
        ks.tiles_left -= 1
        if pai != "?":
            ks.hands[actor].menzen.append(pai)
        # 摸牌后, 该家进入"可打牌"状态; 一发窗口在自家打牌时关闭
        # 岭上标记由前面的杠事件设置

    def _on_dahai(self, e):
        ks = self.ks
        actor = e["actor"]
        pai = e["pai"]
        hand = ks.hands[actor]
        if not e.get("tsumogiri", False) or pai in hand.menzen:
            if pai in hand.menzen:
                hand.menzen.remove(pai)
        # 立直后打出第一张牌 -> 一发窗口结束
        if hand.riichi_accepted:
            hand.ippatsu = False
        ks.last_discarder = actor
        ks.chankan_chance = False
        # 一旦有人打出牌且非第一巡, 破坏双立直可能性
        ks.first_turn_discard[actor] = True

    # --- calls ---
    def _on_chi(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        for c in e["consumed"]:
            if c in hand.menzen:
                hand.menzen.remove(c)
        meld = sorted(e["consumed"] + [e["pai"]])
        hand.chis.append(meld)
        hand.called_chi[tile_id(e["pai"])] += 1
        self._break_ippatsu_and_wriichi()

    def _on_pon(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        for c in e["consumed"]:
            if c in hand.menzen:
                hand.menzen.remove(c)
        meld = sorted(e["consumed"] + [e["pai"]])
        hand.pons.append(meld)
        hand.called_ponkan[tile_id(e["pai"])] += 1
        self._break_ippatsu_and_wriichi()

    def _on_daiminkan(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        for c in e.get("consumed", []):
            if c in hand.menzen:
                hand.menzen.remove(c)
        hand.minkans.append(_deaka(e["pai"]))
        hand.called_ponkan[tile_id(e["pai"])] += 1
        hand.at_rinshan = True
        self._break_ippatsu_and_wriichi()

    def _on_kakan(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        pai = e["pai"]
        if pai in hand.menzen:
            hand.menzen.remove(pai)
        # 把对应 pon 升级为 minkan
        dp = _deaka(pai)
        hand.pons = [m for m in hand.pons if not all(_deaka(x) == dp for x in m)]
        hand.minkans.append(dp)
        hand.called_ponkan[tile_id(pai)] += 1
        hand.at_rinshan = True
        ks.chankan_chance = True
        self._break_ippatsu_and_wriichi()

    def _on_ankan(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        consumed = e.get("consumed", [])
        for c in consumed:
            if c in hand.menzen:
                hand.menzen.remove(c)
        if consumed:
            hand.ankans.append(_deaka(consumed[0]))
            hand.called_ponkan[tile_id(consumed[0])] += 1
        hand.at_rinshan = True
        # 暗杠不破坏一发/双立直(视为无鸣牌), 但按常见规则暗杠不破一发

    def _break_ippatsu_and_wriichi(self):
        ks = self.ks
        ks.any_call_made = True
        for h in ks.hands:
            h.ippatsu = False

    # --- riichi ---
    def _on_reach(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        hand.riichi_declared = True
        # 双立直: 第一巡(自己首次打牌)且此前无人鸣牌
        if not ks.first_turn_discard[actor] and not ks.any_call_made:
            hand.is_w_riichi = True

    def _on_reach_accepted(self, e):
        ks = self.ks
        actor = e["actor"]
        hand = ks.hands[actor]
        hand.riichi_accepted = True
        hand.ippatsu = True

    # --- agari / ryukyoku ---
    def _on_hora(self, e):
        # 由 extract 层在事件后调用 compute_hora_value; 这里只清岭上标记
        pass

    def _on_ryukyoku(self, e):
        pass

    # ============ 算番入口 ============
    def compute_hora_value(self, e: dict):
        """对一个 hora 事件,返回和了者的荣和名义打点(自摸折算)。

        返回 int 点数; 无法算番时抛异常(由上层捕获计入 skip)。
        """
        ks = self.ks
        actor = e["actor"]
        target = e["target"]
        is_ron = actor != target
        hand = ks.hands[actor]

        # winning tile: 荣和=被打出的牌(target 的最后一张弃牌), 自摸=actor 刚摸的牌
        # mjai hora 不带 pai; 荣和牌 = 上一条 dahai 的 pai, 自摸牌 = 上一条 tsumo 的 pai。
        # 由 extract 层把 winning_tile 传进来更稳妥,这里从事件上下文取。
        winning_tile = e["_winning_tile"]

        # 完整门清手牌(含和了牌)
        if is_ron:
            menzen = sorted(hand.menzen + [winning_tile])
        else:
            # 自摸: 和了牌已在 menzen 里(tsumo 时加入)
            menzen = sorted(hand.menzen)

        # additional_hans (偶然役)
        additional = 0
        if hand.riichi_accepted:
            additional += 1
        if hand.is_w_riichi:
            additional += 1
        if hand.ippatsu:
            additional += 1
        if is_ron:
            if ks.tiles_left == 0:
                additional += 1            # 河底捞鱼
            if ks.chankan_chance:
                additional += 1            # 枪杠
        else:
            if hand.is_menzen:
                additional += 1            # 门前清自摸
            if ks.tiles_left == 0 and not hand.at_rinshan:
                additional += 1            # 海底摸月
            if hand.at_rinshan:
                additional += 1            # 岭上开花

        ura = e.get("ura_markers", []) if hand.riichi_accepted else []
        jikaze_idx = (actor - ks.oya) % 4
        jikaze = ["E", "S", "W", "N"][jikaze_idx]

        ron, tsumo_total, han, fu, doras = _agari.calc_agari_point(
            tehai=menzen,
            winning_tile=winning_tile,
            is_ron=is_ron,
            bakaze=WIND_ID[ks.bakaze],
            jikaze=WIND_ID[jikaze],
            is_oya=(actor == ks.oya),
            additional_hans=additional,
            dora_indicators=ks.dora_markers,
            ura_indicators=ura,
            chis=hand.chis,
            pons=hand.pons,
            minkans=hand.minkans,
            ankans=hand.ankans,
            riichi=hand.riichi_accepted,
        )
        # 缓存 dora 张数供 dora 头标签使用
        e["_doras"] = int(doras)
        return ron if is_ron else tsumo_total

    def compute_hora_yaku(self, e: dict):
        """对一个 hora 事件,返回和了者的 12 役 0/1 向量(顺序=value_dist.yaku.YAKU_NAMES)。

        复用与 compute_hora_value 相同的手牌重建。判役在 Python 侧(yaku.judge_yaku)。
        """
        ks = self.ks
        actor = e["actor"]
        target = e["target"]
        is_ron = actor != target
        hand = ks.hands[actor]
        winning_tile = e["_winning_tile"]

        if is_ron:
            menzen = sorted(hand.menzen + [winning_tile])
        else:
            menzen = sorted(hand.menzen)

        jikaze = ["E", "S", "W", "N"][(actor - ks.oya) % 4]
        return judge_yaku(
            menzen=menzen,
            chis=hand.chis,
            pons=hand.pons,
            minkans=hand.minkans,
            ankans=hand.ankans,
            winning_tile=winning_tile,
            is_ron=is_ron,
            bakaze=ks.bakaze,
            jikaze=jikaze,
            is_menzen=hand.is_menzen,
            riichi=hand.riichi_accepted,
            ippatsu=hand.ippatsu,
        )
