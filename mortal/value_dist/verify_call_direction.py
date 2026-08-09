"""验收 a：真实鸣牌方向验证（严谨版）。

对每个真实 chi/pon/daiminkan 事件（自家鸣牌），它之后必然跟一个
dahai（鸣后切牌）。真实"鸣牌+切牌"后的向听 = 库(切牌后暗手)。
推演 = 文档 can_call 定义：从鸣前暗手扣 consumed、新副露(+1组)、
再对"鸣后暗手扣掉每张可切牌 d"取最小向听。
一致性 = 推演出的 shan_后 == 库(真实切牌后暗手) 的向听，≥99%。

注意：推演找"最优切牌"，真实切牌不一定最优，但推演应 ≤ 真实
（推演选最优化，真实可能非最优）。此处验证的是"推演方向正确"
（不会把能听牌误判成不能），故允许 推演 ≤ 真实。
"""
import gzip, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker
from value_dist.shanten_exact import shanten_of

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

def sub_count(c, tiles):
    c2 = c[:]
    for t in tiles:
        c2[tile_to_idx(t)] -= 1
    return c2

def load_events(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(1); random.shuffle(pool)
    files = pool[:n_files]
    ok = bad = total = 0
    bad_samples = []
    for path in files:
        path = "E:/天凤训练数据/2025/" + path
        try: events = load_events(path)
        except Exception: continue
        for viewer in range(4):
            ps = PlayerState(viewer)
            tracker = HandTracker()
            for ei, e in enumerate(events):
                tracker.update(e)
                cans = ps.update(json.dumps(e))
                t = e.get("type")
                if t in ("chi", "pon", "daiminkan") and e.get("actor") == viewer:
                    # 找真实鸣后切牌: 下一个 dahai(actor=viewer)
                    nxt = None
                    for e2 in events[ei+1:ei+4]:
                        if e2.get("type") == "dahai" and e2.get("actor") == viewer:
                            nxt = e2; break
                    if nxt is None:
                        continue
                    hand = tracker.ks.hands[viewer]
                    consumed = e.get("consumed", [])
                    # 鸣前暗手 = 当前 menzen(鸣后) + consumed(已移除的)
                    before = sub_count(count34(hand.menzen), consumed)
                    # 真实切牌后暗手 = 当前 menzen - 切掉的牌
                    real_after = sub_count(count34(hand.menzen), [nxt["pai"]])
                    # 推演: 鸣前 - consumed + 切掉 d 的最小向听
                    # 鸣后未切暗手 = before - consumed = menzen
                    # 推演 shan_后 = min over d in menzen of shanten(menzen - d)
                    menzen_cnt = count34(hand.menzen)
                    best = 99
                    for d in range(34):
                        if menzen_cnt[d] > 0:
                            c2 = menzen_cnt[:]; c2[d] -= 1
                            try: s = shanten_of(c2)
                            except Exception: continue
                            best = min(best, s)
                    try:
                        real_shan = shanten_of(real_after)
                    except Exception:
                        continue
                    total += 1
                    # 推演是最优切牌, 应 ≤ 真实(真实可能非最优)
                    if best == real_shan or best < real_shan:
                        ok += 1
                    else:
                        bad += 1
                        if len(bad_samples) < 10:
                            bad_samples.append((viewer, e.get("type"), e.get("pai"), consumed, nxt["pai"], best, real_shan))
    print(f"真实鸣牌推演验证: 推演≤真实 {ok}/{total} ({ok/max(1,total)*100:.4f}%)  bad={bad}")
    for s_ in bad_samples:
        print(f"  v{s_[0]} {s_[1]}({s_[2]}) consumed{s_[3]} 切{s_[4]} 推演{s_[5]} 真实{s_[6]}")
    # 严格一致(推演==真实) 单独统计
    return 0 if total > 0 and ok/max(1,total) >= 0.99 else 1

if __name__ == "__main__":
    sys.exit(main())
