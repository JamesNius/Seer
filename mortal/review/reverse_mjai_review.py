"""
从 mjai-reviewer 的 JSON 输出中提取原始牌谱事件，支持两种输出格式:
  - mjai 格式 (默认): 每行一个 JSON 事件
  - tenhou/6 格式 (--tenhou): 天凤的 JSON 牌谱格式

用法:
  python reverse_mjai_review.py <review.json> -o <output.json>
  python reverse_mjai_review.py <review.json> --tenhou -o <tenhou.json>

举例:
  python reverse_mjai_review.py review_from_ekyu.json -o raw_events.json
  python reverse_mjai_review.py review_from_ekyu.json --tenhou -o tenhou.json
  cat raw_events.json | python mortal.py 0
"""

import json
import sys
import argparse
import gzip
import re


# ─── mjai tile → tenhou tile id ───────────────────────────────────
# tenhou tile ids: manzu 0-35, pinzu 36-71, souzu 72-107, jihai 108-135
# mjai notation: 1m-9m, 5mr(red5), 1p-9p, 5pr, 1s-9s, 5sr, E,S,W,N,P,F,C
# red fives: 0m=16, 0p=52, 0s=88  (tenhou uses tile 0 for red 5-man)
_TILE_TO_TENHOU = {
    # manzu 1-9
    '1m': 0, '2m': 4, '3m': 8, '4m': 12, '5m': 16, '6m': 20, '7m': 24, '8m': 28, '9m': 32,
    '5mr': 16,  # red 5-man = same as 0m in tenhou
    # pinzu 1-9
    '1p': 36, '2p': 40, '3p': 44, '4p': 48, '5p': 52, '6p': 56, '7p': 60, '8p': 64, '9p': 68,
    '5pr': 52,  # red 5-pin
    # souzu 1-9
    '1s': 72, '2s': 76, '3s': 80, '4s': 84, '5s': 88, '6s': 92, '7s': 96, '8s': 100, '9s': 104,
    '5sr': 88,  # red 5-sou
    # jihai
    'E': 108, 'S': 112, 'W': 116, 'N': 120,  # kaze
    'P': 124, 'F': 128, 'C': 132,             # sangen (白,発,中)
}

_TENHOU_TO_MJAI = {v: k for k, v in _TILE_TO_TENHOU.items()}


def mjai_tile_to_tenhou(tile: str) -> int:
    """mjai 牌表示 → tenhou tile id (0-135)"""
    if tile in ('?', '_'):
        return -1
    return _TILE_TO_TENHOU.get(tile, -1)


def tenhou_tile_to_mjai(tid: int) -> str:
    """tenhou tile id → mjai 牌表示"""
    return _TENHOU_TO_MJAI.get(tid, '?')


def extract_events(review_json_path: str) -> list[dict]:
    """从 mjai-reviewer JSON 中提取原始牌谱事件"""
    with open(review_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 方式1: mjai.ekyu.moe 格式 - 直接取 mjai_log
    if 'mjai_log' in data:
        return data['mjai_log']

    # 方式2: 直接取原始日志
    if 'raw_log' in data:
        return data['raw_log']

    # 方式3: 从 logs 中重建
    events = []
    if 'logs' in data:
        for game_idx, game_log in enumerate(data['logs']):
            if 'game' in game_log:
                game_info = game_log['game']
                start_ev = {
                    'type': 'start_game',
                    'names': game_info.get('names', ['p1', 'p2', 'p3', 'p4']),
                }
                events.append(start_ev)

            kyokus = game_log.get('kyokus', [])
            for kyoku in kyokus:
                if 'start_kyoku' in kyoku:
                    events.append(kyoku['start_kyoku'])
                kyoku_events = kyoku.get('events', [])
                for ev in kyoku_events:
                    clean_ev = {}
                    for k, v in ev.items():
                        if k in ('type', 'actor', 'pai', 'target', 'tsumogiri',
                                 'consumed', 'bakaze', 'dora_marker', 'kyoku',
                                 'honba', 'kyotaku', 'oya', 'scores', 'tehais',
                                 'actor2', 'ten', 'loser', 'sc', 'reason',
                                 'delta_scores', 'uras', 'hora_points',
                                 'nagashi_mangan', 'tenpai'):
                            clean_ev[k] = v
                    if 'type' in clean_ev:
                        events.append(clean_ev)
                if 'end_kyoku' in kyoku:
                    events.append(kyoku['end_kyoku'])
            if 'end_game' in game_log:
                events.append(game_log['end_game'])

    return events


def save_events(events: list[dict], output_path: str, compress: bool = False):
    """保存事件到文件，每行一个 JSON（mjai 格式）"""
    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + '\n')


# ═══════════════════════════════════════════════════════════════
# mjai → tenhou/6 格式转换
# ═══════════════════════════════════════════════════════════════

# tenhou tile IDs (standard tenhou.net/6 encoding)
_TILE_TO_ID = {
    '1m': 11, '2m': 12, '3m': 13, '4m': 14, '5m': 15,
    '6m': 16, '7m': 17, '8m': 18, '9m': 19,
    '1p': 21, '2p': 22, '3p': 23, '4p': 24, '5p': 25,
    '6p': 26, '7p': 27, '8p': 28, '9p': 29,
    '1s': 31, '2s': 32, '3s': 33, '4s': 34, '5s': 35,
    '6s': 36, '7s': 37, '8s': 38, '9s': 39,
    'E': 41, 'S': 42, 'W': 43, 'N': 44,
    'P': 45, 'F': 46, 'C': 47,
    '5mr': 51, '5pr': 52, '5sr': 53,
}
_BAKAZE_IDX = {'E': 0, 'S': 1, 'W': 2, 'N': 3}


def _tid(tile_str: str) -> int:
    """mjai tile string → tenhou tile ID (0 = unknown)"""
    return _TILE_TO_ID.get(tile_str, 0)


def _pad(n: int) -> str:
    return f"{n:02d}"


def _make_chi(actor: int, target: int, pai: str, consumed: list) -> str:
    """tenhou chi naki string: always c<pai><con0><con1>"""
    p = _pad(_tid(pai))
    c0 = _pad(_tid(consumed[0]))
    c1 = _pad(_tid(consumed[1]))
    return f"c{p}{c0}{c1}"


def _make_pon(actor: int, target: int, pai: str, consumed: list) -> str:
    """tenhou pon naki string with direction encoding"""
    p = _pad(_tid(pai))
    c0 = _pad(_tid(consumed[0]))
    c1 = _pad(_tid(consumed[1]))
    rel = (target - actor) % 4
    if rel == 3:   # kamicha
        return f"p{p}{c0}{c1}"
    elif rel == 2: # toimen
        return f"{c0}p{p}{c1}"
    elif rel == 1: # shimocha
        return f"{c0}{c1}p{p}"
    else:
        return f"p{p}{c0}{c1}"


def _make_daiminkan(actor: int, target: int, pai: str, consumed: list) -> str:
    """tenhou daiminkan naki string with direction encoding"""
    p = _pad(_tid(pai))
    c0 = _pad(_tid(consumed[0]))
    c1 = _pad(_tid(consumed[1]))
    c2 = _pad(_tid(consumed[2]))
    rel = (target - actor) % 4
    if rel == 3:   # kamicha
        return f"m{p}{c0}{c1}{c2}"
    elif rel == 2: # toimen
        return f"{c0}m{p}{c1}{c2}"
    elif rel == 1: # shimocha
        return f"{c0}{c1}{c2}m{p}"
    else:
        return f"m{p}{c0}{c1}{c2}"


def _make_kakan(pai: str, consumed: list) -> str:
    """tenhou kakan naki string (default: pon from kamicha)"""
    p = _pad(_tid(pai))
    c0 = _pad(_tid(consumed[0]))
    c1 = _pad(_tid(consumed[1]))
    c2 = _pad(_tid(consumed[2]))
    return f"k{p}{c0}{c1}{c2}"


def _make_ankan(consumed: list) -> str:
    """tenhou ankan naki string: <con0><con1><con2>a<con3>"""
    c0 = _pad(_tid(consumed[0]))
    c1 = _pad(_tid(consumed[1]))
    c2 = _pad(_tid(consumed[2]))
    p  = _pad(_tid(consumed[3]))
    return f"{c0}{c1}{c2}a{p}"


def convert_to_tenhou6(events: list[dict]) -> dict:
    """将 mjai 事件列表转为标准 tenhou.net/6 JSON 格式"""

    names = ['Aさん', 'Bさん', 'Cさん', 'Dさん']
    log = []          # list of kyoku arrays
    current = None    # per-kyoku state
    has_aka = False

    for ev in events:
        t = ev.get('type')

        if t == 'start_game':
            names = ev.get('names', names)
            continue

        if t == 'start_kyoku':
            bakaze = ev.get('bakaze', 'E')
            bakaze_idx = _BAKAZE_IDX.get(bakaze, 0)
            kyoku_in_bakaze = ev.get('kyoku', 1)  # 1..4

            tehais_raw = ev.get('tehais', [[], [], [], []])
            haipai = []
            for pid in range(4):
                hand = [_tid(t) for t in tehais_raw[pid]]
                # detect aka
                for t in tehais_raw[pid]:
                    if t in ('5mr', '5pr', '5sr'):
                        has_aka = True
                haipai.append(hand)

            current = {
                'kyoku_num': bakaze_idx * 4 + kyoku_in_bakaze - 1,
                'honba': ev.get('honba', 0),
                'kyotaku': ev.get('kyotaku', 0),
                'scores': list(ev.get('scores', [25000, 25000, 25000, 25000])),
                'dora_indicators': [_tid(ev.get('dora_marker', '?'))],
                'ura_indicators': [],
                'haipai': haipai,
                'takes': [[], [], [], []],
                'discards': [[], [], [], []],
                'pending_reach': [None, None, None, None],
                'hora_deltas': [],
                'hora_details': [],
            }
            continue

        if current is None:
            continue

        if t in ('end_kyoku', 'end_game'):
            # === build kyoku array ===
            kyoku_arr = [
                [current['kyoku_num'], current['honba'], current['kyotaku']],
                current['scores'],
                current['dora_indicators'],
                current['ura_indicators'],
            ]
            for pid in range(4):
                kyoku_arr.append(current['haipai'][pid])
                kyoku_arr.append(current['takes'][pid])
                kyoku_arr.append(current['discards'][pid])

            # result
            if current['hora_details']:
                result = ["和了"]
                # use first hora's deltas (or combine)
                if current['hora_deltas']:
                    result.append(current['hora_deltas'][0])
                else:
                    result.append([0, 0, 0, 0])
                for detail in current['hora_details']:
                    result.append(detail)
                kyoku_arr.append(result)
            elif current.get('ryukyoku_deltas') is not None:
                kyoku_arr.append(["流局", current['ryukyoku_deltas']])
            else:
                kyoku_arr.append(["流局", [0, 0, 0, 0]])

            log.append(kyoku_arr)
            current = None
            continue

        if t == 'tsumo':
            actor = ev['actor']
            pid = _tid(ev.get('pai', '?'))
            current['takes'][actor].append(pid)

        elif t == 'dahai':
            actor = ev['actor']
            pai_str = ev.get('pai', '?')
            pid = _tid(pai_str)
            tsumogiri = ev.get('tsumogiri', False)

            # check pending reach (reach + dahai → single "r<TT>" entry)
            if current['pending_reach'][actor]:
                discard_id = 60 if tsumogiri else pid
                current['discards'][actor].append(f"r{_pad(discard_id)}")
                current['pending_reach'][actor] = False
            else:
                current['discards'][actor].append(60 if tsumogiri else pid)

        elif t == 'reach':
            actor = ev['actor']
            current['pending_reach'][actor] = True

        elif t == 'dora':
            current['dora_indicators'].append(_tid(ev.get('dora_marker', '?')))

        elif t == 'chi':
            actor = ev['actor']
            target = ev.get('target', (actor + 3) % 4)
            pai = ev.get('pai', '?')
            consumed = ev.get('consumed', [])
            naki = _make_chi(actor, target, pai, consumed)
            current['takes'][actor].append(naki)

        elif t == 'pon':
            actor = ev['actor']
            target = ev.get('target', 0)
            pai = ev.get('pai', '?')
            consumed = ev.get('consumed', [])
            naki = _make_pon(actor, target, pai, consumed)
            current['takes'][actor].append(naki)

        elif t == 'daiminkan':
            actor = ev['actor']
            target = ev.get('target', 0)
            pai = ev.get('pai', '?')
            consumed = ev.get('consumed', [])
            naki = _make_daiminkan(actor, target, pai, consumed)
            current['takes'][actor].append(naki)

        elif t == 'kakan':
            actor = ev['actor']
            pai = ev.get('pai', '?')
            consumed = ev.get('consumed', [])
            naki = _make_kakan(pai, consumed)
            current['discards'][actor].append(naki)

        elif t == 'ankan':
            actor = ev['actor']
            consumed = ev.get('consumed', [])
            naki = _make_ankan(consumed)
            current['discards'][actor].append(naki)

        elif t == 'hora':
            actor = ev['actor']
            target = ev.get('target', actor)
            deltas = ev.get('deltas')
            if deltas:
                current['hora_deltas'].append(deltas)
            current['hora_details'].append([actor, target, target, ""])
            # collect ura markers
            uras = ev.get('ura_markers', [])
            for u in uras:
                uid = _tid(u)
                if uid > 0:
                    current['ura_indicators'].append(uid)

        elif t == 'ryukyoku':
            current['ryukyoku_deltas'] = ev.get('deltas', [0, 0, 0, 0])

    # rule
    rule = {
        'disp': '鳳南喰赤',
    }
    if has_aka:
        rule['aka51'] = 1
        rule['aka52'] = 1
        rule['aka53'] = 1

    return {
        'log': log,
        'name': names,
        'rule': rule,
    }


def save_tenhou6(data: dict, output_path: str):
    """保存为 tenhou/6 JSON 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='从 mjai-reviewer JSON 逆向提取原始牌谱事件'
    )
    parser.add_argument('input', help='mjai-reviewer 输出 JSON 文件')
    parser.add_argument('-o', '--output', default='extracted_events.json',
                        help='输出文件路径 (默认 extracted_events.json)')
    parser.add_argument('-z', '--gzip', action='store_true',
                        help='输出 gzip 压缩')
    parser.add_argument('--tenhou', action='store_true',
                        help='输出 tenhou/6 JSON 格式')
    args = parser.parse_args()

    events = extract_events(args.input)

    if args.tenhou:
        data = convert_to_tenhou6(events)
        save_tenhou6(data, args.output)

        # also save player_id for batch scripts
        with open(args.input, 'r', encoding='utf-8') as f:
            review = json.load(f)
        pid = review.get('player_id', 0)
        pid_file = args.output.replace('.json', '.pid')
        with open(pid_file, 'w') as f:
            f.write(str(pid))

        print(f'转换完成: {len(events)} 事件 → tenhou/6 格式 → {args.output}')
        print(f'共 {len(data["log"])} 局, Player ID: {pid}')
    else:
        output = args.output
        if args.gzip and not output.endswith('.gz'):
            output += '.gz'
        save_events(events, output, compress=args.gzip)
        print(f'提取完成: {len(events)} 个事件 -> {output}')
        print(f'可直接用于: python mortal.py 0 < {output}')


if __name__ == '__main__':
    main()
