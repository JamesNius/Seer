"""闸门4b: dump 100 随机样本 6 头标签供人工目检。

分类整理: 按向听档 / 染手 / dora / value 打点分桶输出。
用法: python -m value_dist.dump_samples [样本数]
"""
import gzip, json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libriichi.state import PlayerState
from value_dist.hand_tracker import HandTracker, tile_id, _deaka
from value_dist.extract import extract_file, N_THRESH


def tile_str(i):
    if i < 9: return f"{i+1}m"
    if i < 18: return f"{i-9+1}p"
    if i < 27: return f"{i-18+1}s"
    return "ESWN PFC".replace(" ", "")[i-27]


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    pool = [f for f in os.listdir("E:/天凤训练数据/2025/") if f.endswith(".mjson")]
    random.seed(6); random.shuffle(pool)
    samples = []
    for path in pool:
        try:
            per_viewer, _ = extract_file("E:/天凤训练数据/2025/" + path)
            for v in range(4):
                samples.extend(per_viewer[v])
        except Exception:
            continue
        if len(samples) >= n_samples:
            break
    samples = samples[:n_samples]
    print(f"dump {len(samples)} 样本")

    # 分类统计
    shan_dist = [0]*5
    somete_pos = 0
    dora_dist = [0]*7
    value_pos = 0
    for s in samples:
        shan = s["shan_label"].argmax(axis=1)
        for p in range(3): shan_dist[shan[p]] += 1
        if s["somete_label"].sum() > 0: somete_pos += 1
        dora = s["dora_label"].argmax(axis=1)
        for p in range(3): dora_dist[dora[p]] += 1
        if s["value_mask"].sum() > 0: value_pos += 1
    print(f"向听档分布(3家): {shan_dist}")
    print(f"染手正例样本: {somete_pos}/{len(samples)}")
    print(f"dora档分布(3家): {dora_dist}")
    print(f"value有标签样本: {value_pos}/{len(samples)}")

    # 打印前 20 个详细
    print("\n=== 前 20 样本详情 ===")
    for i, s in enumerate(samples[:20]):
        shan = s["shan_label"].argmax(axis=1)
        wait_nz = [np.nonzero(s["wait_label"][p])[0].tolist() for p in range(3)]
        dora = s["dora_label"].argmax(axis=1)
        somete = s["somete_label"]
        value = s["value_label"]
        vm = s["value_mask"]
        print(f"[{i}] shan={shan.tolist()} 待牌={[[tile_str(x) for x in w] for w in wait_nz]} "
              f"dora={dora.tolist()} 染手={somete.tolist()} "
              f"value_mask={vm.tolist()} value={[round(x,2) for x in value[0]] if vm[0] else '-'}")


if __name__ == "__main__":
    main()
