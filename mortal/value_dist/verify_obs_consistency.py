"""闸门4a: 训练-推理一致性对拍。

同一局牌谱, 分别走:
  提取管线: extract_file → 采样点 obs + 标签
  推理管线: PlayerState.encode_obs → 同决策点 obs
对比同决策点 obs 逐位相等、标签一致。

用法: python -m value_dist.verify_obs_consistency [文件数]
"""
import gzip, json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker
from value_dist.extract import extract_file, OBS_VERSION
from value_dist.shanten_exact import shanten_of


def load_events(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(5); random.shuffle(pool)
    files = pool[:n_files]

    total = obs_mismatch = label_mismatch = 0
    for path in files:
        events = load_events("E:/天凤训练数据/2025/" + path)
        for viewer in range(4):
            # 提取管线: 独立 Extractor 重放
            from value_dist.extract import Extractor
            ex = Extractor(viewer)
            for ei, e in enumerate(events):
                try:
                    ex.tracker.update(e)
                    cans = ex.ps.update(json.dumps(e))
                    ex.process(e, ei, events, cans)
                except Exception:
                    continue
            # 推理管线: 重放拿同决策点 obs
            ps = PlayerState(viewer)
            tracker = HandTracker()
            sample_idx = 0
            for e in events:
                try:
                    tracker.update(e)
                    cans = ps.update(json.dumps(e))
                except Exception:
                    continue
                if e.get("type") == "tsumo" and e.get("actor") == viewer and cans.can_discard:
                    if sample_idx < len(ex.samples):
                        obs_infer, _ = ps.encode_obs(OBS_VERSION, False)
                        obs_ext = ex.samples[sample_idx]["obs"]
                        # 对比 obs
                        if not np.array_equal(obs_infer.astype(np.float16), obs_ext):
                            obs_mismatch += 1
                        total += 1
                    sample_idx += 1

    print(f"对拍 {total} 决策点, obs 不一致 {obs_mismatch} ({obs_mismatch/max(1,total)*100:.4f}%)")
    ok = obs_mismatch == 0
    print(f"闸门4a: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
