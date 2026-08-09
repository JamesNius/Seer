
import json
import random
import numpy as np
from typing import Optional

from libriichi.state import PlayerState

# ─── tile ↔ action index mapping ───────────────────────────────
_TILES = [
    '1m','2m','3m','4m','5m','6m','7m','8m','9m',
    '1p','2p','3p','4p','5p','6p','7p','8p','9p',
    '1s','2s','3s','4s','5s','6s','7s','8s','9s',
    'E','S','W','N','P','F','C',
    '5mr','5pr','5sr',
]
_TILE_TO_IDX = {t: i for i, t in enumerate(_TILES)}
_IDX_TO_TILE = {i: t for i, t in enumerate(_TILES)}

_TILE_COUNTS = {
    '1m':4,'2m':4,'3m':4,'4m':4,'5m':3,'6m':4,'7m':4,'8m':4,'9m':4,
    '1p':4,'2p':4,'3p':4,'4p':4,'5p':3,'6p':4,'7p':4,'8p':4,'9p':4,
    '1s':4,'2s':4,'3s':4,'4s':4,'5s':3,'6s':4,'7s':4,'8s':4,'9s':4,
    'E':4,'S':4,'W':4,'N':4,'P':4,'F':4,'C':4,
    '5mr':1,'5pr':1,'5sr':1,
}
_ALL_TILES = list(_TILE_COUNTS.keys())

_SUJI_MAP = {
    '1': ['4'], '2': ['5'], '3': ['6'], '4': ['1','7'],
    '5': ['2','8'], '6': ['3','9'], '7': ['4'], '8': ['5'], '9': ['6'],
}

_URA_SUJI = {
    '1': ['2','3'], '2': ['1','3','4'], '3': ['1','2','4','5'],
    '4': ['2','3','5','6'], '5': ['3','4','6','7'],
    '6': ['4','5','7','8'], '7': ['5','6','8','9'],
    '8': ['6','7','9'], '9': ['7','8'],
}

_MATAGI_SUJI = {
    '1': [], '2': [], '3': {'1':['2'], '5':['4'], '7':['6'], '9':['8']}.get,
    '4': {'2':['3'], '6':['5'], '8':['7']}.get,
    '5': {'3':['4'], '7':['6']}.get,
    '6': {'4':['5'], '8':['7']}.get,
    '7': [], '8': [], '9': [],
}

_SEARCH_SAMPLES = 16
_SEARCH_CANDIDATES = 6
_SAFETY_WEIGHT = 3.0


class LightSearch:

    def __init__(self, engine, player_id: int):
        self.engine = engine
        self.player_id = player_id
        self.state = PlayerState(player_id)

        # 对手追踪
        self.opp_rivers = {i: [] for i in range(4)}
        self.opp_calls = {i: [] for i in range(4)}
        self.my_river = []
        self.my_calls = []
        self.dora_markers = []
        self.turn = 0

    # ─── public API ─────────────────────────────────────────────

    def process(self, line: str) -> Optional[str]:
        self._track_event(line)
        cans = self.state.update(line)
        self.turn = self.state.at_turn
        return 'react' if (cans and cans.can_act) else None

    def should_search(self) -> bool:
        return 1 <= self.turn <= 6

    def search(self) -> int:
        obs, mask = self.state.encode_obs(4, False)
        adjusted, _, _ = self._eval(obs, mask)
        return int(np.argmax(adjusted))

    def get_adjusted_q(self) -> list:
        obs, mask = self.state.encode_obs(4, False)
        adjusted, _, original = self._eval(obs, mask)
        m = np.array(mask, dtype=np.bool_)
        return [float(adjusted[i] if m[i] else original[i]) for i in range(len(adjusted))]

    def adjust_reaction(self, reaction_json: str) -> str:
        try:
            reaction = json.loads(reaction_json)
        except json.JSONDecodeError:
            return reaction_json
        if 'meta' not in reaction:
            return reaction_json
        meta = reaction['meta']
        adjusted_q = self.get_adjusted_q()
        if 'q_values' in meta:
            meta['original_q'] = meta['q_values']
            meta['search_adjusted'] = True
        meta['q_values'] = adjusted_q
        return json.dumps(reaction, ensure_ascii=False)

    def build_reaction(self, action_idx: int, cans) -> str:
        if action_idx <= 36:
            tile = self._action_to_tile(action_idx)
            tsumo = self.state.last_self_tsumo()
            tsumogiri = (tsumo == tile) if tsumo else False
            return json.dumps({
                'type': 'dahai', 'actor': self.player_id,
                'pai': tile, 'tsumogiri': tsumogiri,
            })
        elif action_idx == 37:
            return json.dumps({'type': 'reach', 'actor': self.player_id})
        elif action_idx == 43:
            return json.dumps({'type': 'hora', 'actor': self.player_id, 'target': self.player_id})
        else:
            return '{"type":"none"}'


    def _eval(self, obs, mask):
        _, q_values, _, _ = self.engine.react_batch([obs], [mask], None)
        q = np.array(q_values[0], dtype=np.float32)
        m = np.array(mask, dtype=np.bool_)
        pool = self._build_pool()

        opp_hands = {}
        for pid in range(4):
            if pid == self.player_id:
                continue
            opp_hands[pid] = self._sample_hands(pid, pool, _SEARCH_SAMPLES)

        safety = np.zeros(q.shape, dtype=np.float32)
        for action in range(37):
            if not m[action]:
                continue
            tile = self._action_to_tile(action)
            safety[action] = self._tile_danger_v2(tile, opp_hands)

        q_masked = np.where(m, q, -1e10)
        adjusted = q_masked + safety * _SAFETY_WEIGHT

        return adjusted, safety, q_masked


    def _build_pool(self) -> dict:
        pool = dict(_TILE_COUNTS)

        def remove(t):
            if t in pool and pool[t] > 0:
                pool[t] -= 1

        for t in self._get_my_hand():
            remove(t)
        for t in self.my_river:
            remove(t)
        for call in self.my_calls:
            for t in call.get('consumed', []):
                remove(t)

        for pid in range(4):
            if pid == self.player_id:
                continue
            for t in self.opp_rivers[pid]:
                remove(t)
            for call in self.opp_calls[pid]:
                for t in call.get('consumed', []):
                    remove(t)

        for dora in self.dora_markers:
            remove(dora.rstrip('r'))

        return pool

    def _get_my_hand(self) -> list:
        tiles = []
        the_hai = self.state.tehai
        if hasattr(the_hai, '__getitem__') and hasattr(the_hai, '__len__'):
            for i in range(min(len(the_hai), 34)):
                try:
                    count = int(the_hai[i])
                except (TypeError, IndexError):
                    count = 0
                if count > 0:
                    tiles.extend([_TILES[i]] * count)
        return tiles

    def _sample_hands(self, pid: int, pool: dict, n: int) -> list:
        candidates = []
        for t, c in pool.items():
            if c > 0:
                candidates.extend([t] * c)
        if len(candidates) < 13:
            return []

        river = self.opp_rivers[pid]
        suit_discards = {'m': 0, 'p': 0, 's': 0, 'z': 0}
        for t in river:
            if t[-1] in 'mps':
                suit_discards[t[-1]] += 1
            else:
                suit_discards['z'] += 1

        total_discards = sum(suit_discards.values()) or 1
        suit_weights = {}
        for suit in 'mpsz':
            discard_ratio = suit_discards[suit] / total_discards
            suit_weights[suit] = max(0.2, 1.0 - discard_ratio)

        hands = []
        for _ in range(n):
            weighted = []
            for t in candidates:
                w = suit_weights.get(t[-1], 1.0) if t[-1] in 'mps' else suit_weights.get('z', 1.0)
                if t in river:
                    w *= 0.1
                weighted.extend([t] * max(1, int(w * 10)))

            if len(weighted) < 13:
                weighted = list(candidates)

            sample = random.sample(weighted, 13)
            hands.append(sample)

        return hands

    def _tile_danger_v2(self, tile: str, opp_hands: dict) -> float:
        score = 0.0
        tile_base = tile.rstrip('r')
        is_honor = tile_base in ('E','S','W','N','P','F','C')
        is_terminal = tile_base[0] in '19' if tile_base[0].isdigit() else False

        for pid, hands in opp_hands.items():
            visible = set(self.opp_rivers[pid])
            for call in self.opp_calls[pid]:
                visible.update(call.get('consumed', []))

            if tile in visible or tile_base in visible:
                score += 0.8
                continue

            if tile_base[0].isdigit():
                num = tile_base[0]
                suit = tile_base[1]
                for safe_tile in visible:
                    if safe_tile[0] == num and safe_tile[1] != suit:
                        continue  # 不同花色不算
                    if safe_tile[0].isdigit() and safe_tile[0] in _SUJI_MAP.get(num, []):
                        # same suit suji
                        if safe_tile[1] == suit:
                            score += 0.25
                            break

            useful_count = 0
            total_hands = len(hands)
            for hand in hands:
                if self._tile_useful_to_hand(tile_base, hand):
                    useful_count += 1

            if total_hands > 0:
                useful_ratio = useful_count / total_hands
                if useful_ratio > 0.5:
                    score -= 0.5
                elif useful_ratio > 0.25:
                    score -= 0.25

            if tile_base[0] in '456':
                score -= 0.15

            if tile_base[0] in '34567' and not is_terminal:
                no_suji_penalty = True
                for safe_tile in visible:
                    if safe_tile[0].isdigit() and tile_base[1] == safe_tile[1]:
                        if safe_tile[0] in _SUJI_MAP.get(tile_base[0], []):
                            no_suji_penalty = False
                            break
                if no_suji_penalty:
                    score -= 0.2

            if is_honor:
                seen = sum(
                    r.count(tile_base) for r in self.opp_rivers.values()
                ) + sum(
                    c.get('consumed', []).count(tile_base)
                    for calls in self.opp_calls.values() for c in calls
                )
                if seen >= 2:
                    score += 0.6
                elif seen == 1:
                    score += 0.2

        return max(-3.0, min(3.0, score))

    def _tile_useful_to_hand(self, tile: str, hand: list) -> bool:
        if not tile or not isinstance(tile, str):
            return False

        is_honor = tile in ('E','S','W','N','P','F','C')
        suit = tile[1] if len(tile) >= 2 and tile[1] in 'mps' else ('z' if is_honor else None)
        if suit is None:
            return False

        num = int(tile[0]) if tile[0].isdigit() else None

        if suit == 'z':
            return tile in hand

        for h in hand:
            if not h or not isinstance(h, str) or len(h) < 2:
                continue
            if h == tile:
                return True
            if h[1] == suit and h[0].isdigit() and num is not None:
                hn = int(h[0])
                diff = abs(num - hn)
                if 1 <= diff <= 2:
                    return True
        return False

    # ─── helpers ─────────────────────────────────────────────────

    def _track_event(self, line: str):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        t = ev.get('type')
        actor = ev.get('actor')
        if actor is None:
            return

        if t == 'dahai':
            pai = ev.get('pai', '?')
            if actor == self.player_id:
                self.my_river.append(pai)
            elif 0 <= actor <= 3:
                self.opp_rivers[actor].append(pai)
        elif t in ('pon', 'chi', 'daiminkan', 'kakan', 'ankan'):
            if actor == self.player_id:
                self.my_calls.append(ev)
            elif 0 <= actor <= 3:
                self.opp_calls[actor].append(ev)
        elif t == 'dora':
            self.dora_markers.append(ev.get('dora_marker', '?'))

    def _action_to_tile(self, action_idx: int) -> str:
        return _IDX_TO_TILE.get(action_idx, '?')

    @property
    def at_turn(self) -> int:
        return self.state.at_turn
