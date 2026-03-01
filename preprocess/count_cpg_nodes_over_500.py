import argparse, os, re, glob

NODE_LINE = re.compile(r'^\s*("?[^"\s\[]+"?)\s*\[', re.ASCII)

def count_nodes_in_dot(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "->" in line:        # edge line
                continue
            if line.lstrip().startswith(("graph ", "node ", "edge ")):
                continue
            if NODE_LINE.match(line) and "label" in line:
                n += 1
    return n

def stat_dir(dot_dir: str, threshold: int = 500):
    dot_files = sorted(glob.glob(os.path.join(dot_dir, "**", "*.dot"), recursive=True))
    total = 0
    big = 0
    for p in dot_files:
        c = count_nodes_in_dot(p)
        total += 1
        if c > threshold:
            big += 1
    ratio = (big / total) if total else 0.0
    return total, big, ratio

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--ffmpeg_qemu", default="dots-cpg/ffmpeg-qemu")
    # ap.add_argument("--chrome_debian", default="dots-cpg/chrome-debian")
    # ap.add_argument("--bigvul", default="dots-cpg/big-vul")
    ap.add_argument("--primevul", default="dots-cpg/chrome-debian")
    ap.add_argument("--thr", type=int, default=500)
    args = ap.parse_args()

    for name, d in [
        ("PrimeVul", args.primevul)
    ]:
        if not os.path.isdir(d):
            print(f"{name}: directory not found -> {d}")
            continue
        total, big, ratio = stat_dir(d, args.thr)
        print(f"{name}: total={total}, >{args.thr}={big}, ratio={ratio:.4%}")
