"""端到端验证: 起 review server, 发一份完整谱, 检查每个决策条目带 3 家 value_ge 且单调。

用法: python -m value_dist.e2e_test <mjson路径> [player_id]
需先在另一终端启动 mortal_review_server.py, 或让本脚本自动起子进程。
"""
import gzip, json, socket, subprocess, sys, os, time, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from value_dist.model import THRESHOLDS

PORT = 9877  # 用非默认端口, 避免与常驻服务冲突


def read_log(path):
    op = gzip.open if path.endswith(".mjson") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def run_server_proc():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.Popen(
        [sys.executable, "mortal_review_server.py", str(PORT)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    path = sys.argv[1]
    player_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    lines = read_log(path)
    print(f"谱 {os.path.basename(path)}  共 {len(lines)} 行  player_id={player_id}")

    proc = run_server_proc()
    try:
        # 等服务起来
        for _ in range(60):
            try:
                s = socket.create_connection(("127.0.0.1", PORT), timeout=2)
                s.close(); break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("server 未启动")

        sock = socket.create_connection(("127.0.0.1", PORT), timeout=300)
        wf = sock.makefile("w", encoding="utf-8")
        rf = sock.makefile("r", encoding="utf-8")
        wf.write(f"{player_id}\n")
        for ln in lines:
            wf.write(ln + "\n")
        wf.flush()
        sock.shutdown(socket.SHUT_WR)

        reactions = [json.loads(l) for l in rf if l.strip()]
        sock.close()
    finally:
        proc.terminate()

    print(f"收到 {len(reactions)} 条反应")
    # 最后一条是汇总 {model_tag, phi_matrix, has_oracle, has_value_ge}
    summary = reactions[-1]
    print("summary keys:", list(summary.keys()), "has_value_ge:", summary.get("has_value_ge"))

    n_decision = n_mono_bad = 0
    sample_shown = 0
    FIELDS = ("value_ge", "yaku", "call_ponkan", "call_chi", "dora")
    for r in reactions[:-1]:
        meta = r.get("meta")
        if not isinstance(meta, dict):
            continue
        # 决策条目 = oracle 带四头 或 reading
        per_player = None
        oracle = meta.get("oracle")
        if oracle and isinstance(oracle, list) and oracle[0] and "value_ge" in oracle[0][0]:
            per_player = oracle[0]     # 主决策样本的 3 家
        elif "reading" in meta:
            per_player = meta["reading"][0]
        if per_player is None:
            continue
        n_decision += 1
        # 字段完整性 + value_ge 单调性
        for k in range(3):
            p = per_player[k]
            for f in ("value_ge", "yaku", "call_ponkan", "call_chi", "dora"):
                assert f in p, f"缺字段 {f}"
            assert len(p["value_ge"]) == 6 and len(p["yaku"]) == 14
            assert len(p["call_ponkan"]) == 34 and len(p["call_chi"]) == 34
            arr = p["value_ge"]
            if any(arr[i] < arr[i+1] - 1e-6 for i in range(5)):
                n_mono_bad += 1
        if sample_shown < 3:
            p0 = per_player[0]
            print(f"  样例下家: value_ge={[round(x,3) for x in p0['value_ge']]}")
            print(f"          dora={p0['dora']} yaku_top={sorted(enumerate(p0['yaku']),key=lambda x:-x[1])[:3]}")
            pk = p0['call_ponkan']; top_call = sorted(enumerate(pk),key=lambda x:-x[1])[:3]
            print(f"          call_ponkan_top3(tile,prob)={top_call}")
            sample_shown += 1

    print(f"\n决策条目带四头: {n_decision}")
    print(f"value_ge 单调性违规: {n_mono_bad}")
    assert n_decision > 0, "没有任何决策条目带四头!"
    assert n_mono_bad == 0, "存在单调性违规!"
    print("端到端验证通过 ✅")


if __name__ == "__main__":
    main()
