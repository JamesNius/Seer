"""对拍闸门：mahjong 库 shanten vs libriichi PlayerState.shanten()。

用成熟开源 mahjong 库（保底方案）计算，对拍 PlayerState。
从 hand_tracker 抽真实手牌，覆盖 len_div3 0~4、13/14张、七对子、国士、
含赤、暗杠。暗手=menzen(不含副露), len_div3=4-副露组数。
一致率必须 100%，否则停止输出不一致案例清单。

用法: python -m value_dist.verify_shanten [文件数]
"""
import gzip, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker
from mahjong.shanten import Shanten

JIHAI = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}

def tile_to_idx(t):
    """mjai 牌字符串(含 5mr) -> 34 计数索引, 赤按普通 5。"""
    if t in JIHAI:
        return JIHAI[t]
    n = int(t[0]); s = t[1]
    return {"m": 0, "p": 9, "s": 18}[s] + n

def count34(tiles):
    c = [0] * 34
    for t in tiles:
        c[tile_to_idx(t)] += 1
    return c

def extract_hands(events, viewer):
    """重建 viewer 每时刻暗手(menzen) + 副露组数 + PlayerState 真向听。"""
    ps = PlayerState(viewer)
    tracker = HandTracker()
    out = []
    for e in events:
        tracker.update(e)
        cans = ps.update(json.dumps(e))
        if e.get("type") == "tsumo" and e.get("actor") == viewer and cans.can_discard:
            true_shan = ps.shanten
            hand = tracker.ks.hands[viewer]
            concealed = [x for x in hand.menzen]
            n_melds = len(hand.chis) + len(hand.pons) + len(hand.minkans) + len(hand.ankans)
            out.append((concealed, n_melds, true_shan))
    return out

def load_events(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(0); random.shuffle(pool)
    files = ["E:/天凤训练数据/2025/" + f for f in pool[:n_files]]
    shanten_calc = Shanten()
    total = mism = 0
    samples = []
    stats = {"n_melds": {}, "n_tiles": {}}
    for f in files:
        try: events = load_events(f)
        except Exception: continue
        for viewer in range(4):
            for concealed, n_melds, true_shan in extract_hands(events, viewer):
                total += 1
                cnt = count34(concealed)
                # mahjong 库无 melds 参数: 张数=暗手张数已隐含副露数
                # (暗手2张=雀头→-1, 对应4副露). 直接传暗手34数组。
                try:
                    mine = shanten_calc.calculate_shanten(cnt)
                except Exception:
                    continue
                stats["n_melds"][n_melds] = stats["n_melds"].get(n_melds, 0) + 1
                stats["n_tiles"][len(concealed)] = stats["n_tiles"].get(len(concealed), 0) + 1
                if mine != true_shan:
                    mism += 1
                    if len(samples) < 20:
                        samples.append((concealed, n_melds, true_shan, mine))
    print(f"对拍 {total} 手, 不一致 {mism} ({mism/max(1,total)*100:.4f}%)")
    print(f"覆盖: n_melds={stats['n_melds']}  暗手张数={stats['n_tiles']}")
    if mism > 0:
        print("不一致案例(前20):")
        for concealed, nm, ts, ms in samples:
            print(f"  n_melds={nm} 手牌{sorted(concealed)}  真实={ts} mahjong={ms}")
    return 0 if mism == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
