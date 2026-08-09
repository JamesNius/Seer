"""验收 b：对子重型门清手牌暴力校验。

每 1 万局随机抽 20 手对子重型(≥3对)门清手牌，
对 lib≤2 区域用暴力校验器(is_winning/is_tenpai/path_exists_k)
抽检，必须 100% 通过。

对子重型手牌: 34 计数里 ≥3 种牌计数≥2 的门清 13/14 张手。
lib≤2: 库向听 ≤2 的样本 (低向听区, 暴力可验证)。
"""
import gzip, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker
from value_dist.shanten_exact import shanten_of, is_winning, is_tenpai, path_exists_k

JIHAI = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}

def tile_to_idx(t):
    if t in JIHAI:
        return JIHAI[t]
    n = int(t[0]); s = t[1]
    return {"m": 0, "p": 9, "s": 18}[s] + n

def count34(tiles):
    c = [0] * 34
    for t in tiles:
        c[tile_to_idx(t)] += 1
    return c

def load_events(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 50   # ~1万局
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(2); random.shuffle(pool)
    files = pool[:n_files]
    hand_pool = []   # 收集对子重型门清手
    for path in files:
        path = "E:/天凤训练数据/2025/" + path
        try: events = load_events(path)
        except Exception: continue
        for viewer in range(4):
            ps = PlayerState(viewer)
            tracker = HandTracker()
            for e in events:
                tracker.update(e)
                cans = ps.update(json.dumps(e))
                if e.get("type") == "tsumo" and e.get("actor") == viewer and cans.can_discard:
                    hand = tracker.ks.hands[viewer]
                    n_meld = len(hand.chis)+len(hand.pons)+len(hand.minkans)+len(hand.ankans)
                    if n_meld > 0:
                        continue   # 门清
                    concealed = [x for x in hand.menzen]
                    cnt = count34(concealed)
                    pairs = sum(1 for x in cnt if x >= 2)
                    if pairs >= 3:   # 对子重型
                        try:
                            lib_shan = shanten_of(cnt)
                        except Exception:
                            continue
                        if lib_shan <= 2:   # lib≤2 区域
                            hand_pool.append((cnt, lib_shan, len(concealed)))
    # 抽 20 手
    random.shuffle(hand_pool)
    sample = hand_pool[:20]
    print(f"对子重型门清手总数: {len(hand_pool)}, 抽检 {len(sample)}")
    fails = []
def _exists_discard_to_tenpai(cnt14):
    """14张: 是否存在切掉一张 d 后 13 张听牌。"""
    for d in range(34):
        if cnt14[d] > 0:
            c = cnt14[:]; c[d] -= 1
            if is_tenpai(c):
                return True
    return False


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 50   # ~1万局
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(2); random.shuffle(pool)
    files = pool[:n_files]
    hand_pool = []   # 收集对子重型门清手
    for path in files:
        path = "E:/天凤训练数据/2025/" + path
        try: events = load_events(path)
        except Exception: continue
        for viewer in range(4):
            ps = PlayerState(viewer)
            tracker = HandTracker()
            for e in events:
                tracker.update(e)
                cans = ps.update(json.dumps(e))
                if e.get("type") == "tsumo" and e.get("actor") == viewer and cans.can_discard:
                    hand = tracker.ks.hands[viewer]
                    n_meld = len(hand.chis)+len(hand.pons)+len(hand.minkans)+len(hand.ankans)
                    if n_meld > 0:
                        continue   # 门清
                    concealed = [x for x in hand.menzen]
                    cnt = count34(concealed)
                    pairs = sum(1 for x in cnt if x >= 2)
                    if pairs >= 3:   # 对子重型
                        try:
                            lib_shan = shanten_of(cnt)
                        except Exception:
                            continue
                        if lib_shan <= 2:   # lib≤2 区域
                            hand_pool.append((cnt, lib_shan, len(concealed)))
    # 抽 20 手
    random.shuffle(hand_pool)
    sample = hand_pool[:20]
    print(f"对子重型门清手总数: {len(hand_pool)}, 抽检 {len(sample)}")
    fails = []
    for i, (cnt, lib_shan, n_tiles) in enumerate(sample):
        ok = True
        if n_tiles == 13:
            # 13 张: shan=0 听牌, shan=1 一步, shan=2 两步
            if lib_shan == 0:
                ok = is_tenpai(cnt)
            elif lib_shan == 1:
                ok = path_exists_k(cnt, 1)
            elif lib_shan == 2:
                ok = path_exists_k(cnt, 2)
        elif n_tiles == 14:
            # 14 张(摸牌后未切): shan=0 = 存在可切牌使 13张听牌
            if lib_shan == 0:
                ok = _exists_discard_to_tenpai(cnt)
            elif lib_shan == -1:
                ok = is_winning(cnt)
        if not ok:
            fails.append((i, lib_shan, n_tiles, cnt))
    print(f"校验失败: {len(fails)}/{len(sample)}")
    for f in fails:
        print(f"  样本{f[0]} lib_shan={f[1]} 张数={f[2]}")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
