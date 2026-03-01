import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========== 参数设置 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='PrimeVul.json', type=str, help="dataset name")
parser.add_argument("--workers", default=max(1, (os.cpu_count() or 2) // 2), type=int,
                    help="parallel workers (joern is heavy; start with 2~4)") # 设置进程数，一般不宜过多，2~4 是个不错的起点
args = parser.parse_args()

# 路径配置（这些是“根目录”）
dataset_file = Path(f"../data/{args.dataset}")
source_root = Path("source")
if args.dataset == 'FFmpeg-Qemu.json':
    dot_output_dir = Path("dots-cpg/ffmpeg-qemu")
elif args.dataset == 'PrimeVul.json':
    dot_output_dir = Path("dots-cpg/primevul")
elif args.dataset == 'Chrome-Debian.json':
    dot_output_dir = Path("dots-cpg/chrome-debian")
joern_workspace = Path("workspace")

# 创建目录
source_root.mkdir(parents=True, exist_ok=True)
dot_output_dir.mkdir(parents=True, exist_ok=True)
joern_workspace.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd_list):
    """运行命令并返回 (ok, stdout, stderr)"""
    r = subprocess.run(cmd_list, capture_output=True, text=True)
    return (r.returncode == 0, r.stdout, r.stderr)


def process_one(task):
    """
    task: (idx, code, label)
    返回 (idx, ok, msg)
    """
    idx, code, label = task

    # 每个样本独立目录，避免并行冲突
    c_source_dir = source_root / f"{idx}"
    c_source_dir.mkdir(parents=True, exist_ok=True)

    c_file = c_source_dir / f"{label}.c"
    with open(c_file, "w", encoding="utf-8") as cf:
        cf.write(code)

    cpg_bin_path = joern_workspace / f"{idx}-cpg-{label}.bin"
    dot_tmp_dir = dot_output_dir / f"{idx}"   # joern-export 的 out 目录（临时）
     # joern-export 要求 out 目录不存在：先删掉旧的（可能是上次残留）
    if dot_tmp_dir.exists():
        shutil.rmtree(dot_tmp_dir, ignore_errors=True)

    ok, _, err = run_cmd(["joern-export", str(cpg_bin_path), "--repr", "cpg14", "--out", str(dot_tmp_dir)])

    try:
        # joern-parse
        ok, _, err = run_cmd(["joern-parse", str(c_source_dir), "-o", str(cpg_bin_path)])
        if not ok:
            return (idx, False, f"[!] joern-parse failed: {c_file}\n{err}")

        # joern-export
        ok, _, err = run_cmd(["joern-export", str(cpg_bin_path), "--repr", "cpg14", "--out", str(dot_tmp_dir)])
        if not ok:
            return (idx, False, f"[!] joern-export failed: {c_file}\n{err}")

        generated_dot = dot_tmp_dir / "0-cpg.dot"
        if generated_dot.exists():
            renamed_dot = dot_output_dir / f"{idx}-cpg-{label}.dot"
            # 若目标存在，先删掉避免 rename 报错
            if renamed_dot.exists():
                renamed_dot.unlink()
            generated_dot.rename(renamed_dot)
            return (idx, True, f"[+] {c_file} --> {renamed_dot}")
        else:
            return (idx, False, f"[!] Dot file not found: {c_file} (expected {generated_dot})")

    finally:
        # 清理临时目录
        if dot_tmp_dir.exists():
            shutil.rmtree(dot_tmp_dir, ignore_errors=True)
        # 清理 source 和 cpg bin（可选）：
            shutil.rmtree(c_source_dir, ignore_errors=True)
            if cpg_bin_path.exists(): cpg_bin_path.unlink()


# ========== 读取 JSON 文件 ==========
with open(dataset_file, "r", encoding="utf-8") as f:
    if args.dataset == 'PrimeVul.json':
        data = [json.loads(line) for line in f if line.strip()]   # JSONL
    elif args.dataset == 'Chrome-Debian.json':
        data = [json.loads(line) for line in f if line.strip()]   # JSONL
    elif args.dataset == 'FFmpeg-Qemu.json':
        data = json.load(f)                                       # 单个 JSON
    else:
        raise ValueError(f"Unknown dataset format for {args.dataset}")

tasks = [(idx, item["func"], item["target"]) for idx, item in enumerate(data)]

print(f"[*] Total samples: {len(tasks)}, workers: {args.workers}")

# ========== 并行执行 ==========
ok_cnt = 0
fail_cnt = 0

with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(process_one, t) for t in tasks]
    for fu in as_completed(futures):
        idx, ok, msg = fu.result()
        print(msg, flush=True)
        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1

print(f"\n✅ Done. success={ok_cnt}, failed={fail_cnt}")