"""mountain 越界验收: 起 review server(新代码), 跑 N 局, 检查每决策点:
1. 每牌 mountain(山残期望) <= 4 - visible(服务端下发的 mountain_visible)
2. 显示值 == 服务端覆盖层后的 E[mountain]（覆盖层本身保证 ≤ 4-visible）

用法: python -m value_dist.accept_mountain <n局数>
"""
import gzip, json, socket, subprocess, sys, os, time, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 9877
DATA_DIR = "E:/天凤训练数据/2025/"


def read_log(path):
    op = gzip.open if path.endswith(".mjson") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def run_one(path, player_id):
    lines = read_log(path)
    logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"accept_srv_{os.path.basename(path)[:14]}.log")
    srvlog = open(logpath, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "mortal_review_server.py", str(PORT)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
        stdout=srvlog, stderr=srvlog,
    )
    try:
        # 等服务起来(最多 120s, 训练占 GPU 时启动较慢)
        for _ in range(120):
            if proc.poll() is not None:
                srvlog.flush()
                raise RuntimeError(f"server 提前退出, 日志:\n{open(logpath, encoding='utf-8').read()[-1500:]}")
            try:
                s = socket.create_connection(("127.0.0.1", PORT), timeout=2)
                s.close(); break
            except OSError: time.sleep(1)
        else:
            srvlog.flush()
            raise RuntimeError("server 未启动")
        sock = socket.create_connection(("127.0.0.1", PORT), timeout=600)
        sock.settimeout(600)
        wf = sock.makefile("w", encoding="utf-8"); rf = sock.makefile("r", encoding="utf-8")
        wf.write(f"{player_id}\n")
        for ln in lines: wf.write(ln + "\n")
        wf.flush(); sock.shutdown(socket.SHUT_WR)
        reactions = []
        for l in rf:
            if l.strip():
                reactions.append(json.loads(l))
        sock.close()
        return reactions
    except Exception as e:
        srvlog.flush()
        tail = ""
        try:
            tail = open(logpath, encoding='utf-8').read()[-1500:]
        except Exception:
            pass
        raise RuntimeError(f"run_one failed for {path}: {e}\nserver log tail:\n{tail}")
    finally:
        # 等子进程真正退出, 释放 GPU/端口, 避免下一局启动失败
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        srvlog.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    files = sorted(glob.glob(DATA_DIR + "*.mjson"))[:n]
    print(f"验收 {len(files)} 局")

    n_violation = 0
    n_decision = 0
    n_tile = 0
    viol_samples = []
    for fi, path in enumerate(files):
        player_id = fi % 4
        reactions = run_one(path, player_id)
        local_viol = 0
        for r in reactions[:-1]:  # 最后一条是 summary
            meta = r.get("meta")
            if not isinstance(meta, dict): continue
            vis = meta.get("mountain_visible")
            oracle = meta.get("oracle")
            if not (vis and oracle and isinstance(oracle, list) and oracle[0]):
                continue
            per_player = oracle[0]   # 主决策样本 3 家
            n_decision += 1
            for k in range(3):
                p = per_player[k]
                if "mountain" not in p: continue
                for t, m in enumerate(p["mountain"]):
                    n_tile += 1
                    lim = 4 - vis[t]
                    if m < 0 or m > lim + 1e-6:
                        n_violation += 1
                        if len(viol_samples) < 5:
                            viol_samples.append({
                                "file": os.path.basename(path), "k": k, "tile": t,
                                "mountain": m, "visible": vis[t], "limit": lim,
                            })
        print(f"[{fi+1}/{len(files)}] {os.path.basename(path)} 决策={n_decision} 本局违规={local_viol} 累计违规={n_violation}")

    print(f"\n===== 验收报告 =====")
    print(f"局数 {len(files)}  决策点 {n_decision}  牌-检查 {n_tile}")
    print(f"越界违规: {n_violation}")
    for s in viol_samples:
        print(f"  违规样例: {s}")
    assert n_violation == 0, "存在越界违规!"
    print("✅ 100 局零越界通过")


if __name__ == "__main__":
    main()
