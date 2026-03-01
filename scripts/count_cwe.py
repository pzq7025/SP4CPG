import os
import re
import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CWE_REGEX = re.compile(r'\bCWE[-\s:]*0*([0-9]{1,5})\b', flags=re.IGNORECASE)
SPLITS = ["train_unpaired", "valid_unpaired", "test_unpaired"]

def extract_cwe_from_value(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        for v in val:
            r = extract_cwe_from_value(v)
            if r:
                return r
        return None
    if isinstance(val, dict):
        for k in ('cwe','cwe_id','CWE','metadata'):
            if k in val:
                r = extract_cwe_from_value(val[k])
                if r:
                    return r
        return None
    if isinstance(val, int):
        return f"CWE-{val}"
    if not isinstance(val, str):
        return None
    m = CWE_REGEX.search(val)
    if m:
        return f"CWE-{int(m.group(1))}"
    s = val.strip()
    if s.isdigit():
        return f"CWE-{int(s)}"
    m2 = re.search(r'\bcwe[:\s#-]*([0-9]{1,5})\b', val, flags=re.IGNORECASE)
    if m2:
        return f"CWE-{int(m2.group(1))}"
    return None

def extract_cwe_from_example(example):
    CANDIDATE_FIELDS = ['cwe', 'cwe_id', 'CWE', 'CWE_ID', 'CWE-ID', 'cwe_ids', 'vuln_cwe', 'metadata', 'labels', 'label']
    for f in CANDIDATE_FIELDS:
        if f in example:
            r = extract_cwe_from_value(example[f])
            if r:
                return r
    for k, v in example.items():
        if isinstance(v, str):
            r = extract_cwe_from_value(v)
            if r:
                return r
    return None

def is_vulnerable_example(example):
    # 检查常见字段判断是否为漏洞样本（支持 bool / int / 字符串）
    for key in ("is_vulnerable", "vulnerable", "is_vul", "bug", "label_vuln", "vuln"):
        if key in example:
            v = example[key]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "1", "vulnerable", "vuln"):
                    return True
                if s in ("false", "no", "0", "benign", "clean"):
                    return False
    # 如果有字段 "label" 且值为字符串或数字，尝试识别（一些 split 用 label=1 表示 vuln）
    if "label" in example:
        v = example["label"]
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "vulnerable", "vuln"):
                return True
            if s in ("0", "false", "benign"):
                return False
    # 未能判断 -> 认为不是 vuln（可改为返回 None 表示未知）
    return False

def compute_vulnerable_cwe_counts(dataset_id="ASSERT-KTH/PrimeVul", splits=None):
    from datasets import load_dataset
    if splits is None:
        splits = SPLITS
    combined = Counter()
    total_checked = 0
    total_vuln = 0
    for s in splits:
        try:
            ds = load_dataset(dataset_id, split=s)
        except Exception as e:
            # 有些数据集在某些环境下需要使用 streaming 或不同的加载方式
            # 尝试通过加载全部然后取 split 字段
            try:
                ds_all = load_dataset(dataset_id)
                ds = ds_all[s]
            except Exception:
                raise
        for ex in ds:
            total_checked += 1
            if not is_vulnerable_example(ex):
                continue
            total_vuln += 1
            c = extract_cwe_from_example(ex)
            if c is None:
                combined["UNKNOWN"] += 1
            else:
                combined[c] += 1
    return combined, total_checked, total_vuln

def plot_all_counts(counter, out_png="cwe_vulnerable_distribution_all.png"):
    if not counter:
        raise ValueError("无数据可绘制")
    df = pd.DataFrame(list(counter.items()), columns=["CWE", "count"])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    n = len(df)
    width = max(12, int(0.35 * n))
    plt.figure(figsize=(width, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(data=df, x="CWE", y="count", palette="viridis")
    ax.set_xlabel("CWE")
    ax.set_ylabel("样本数（is_vulnerable=True）")
    ax.set_title("每一种 CWE 的漏洞样本数量分布（仅 is_vulnerable=True）")
    plt.xticks(rotation=90, fontsize=8)
    for p in ax.patches:
        h = int(p.get_height())
        if h > 0:
            ax.annotate(str(h), (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=7, xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved full CWE vulnerable distribution plot to {out_png}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="统计每一种CWE类型的漏洞样本数量（is_vulnerable=True）")
    parser.add_argument("--dataset", "-d", default="ASSERT-KTH/PrimeVul", help="HuggingFace 数据集 id")
    parser.add_argument("--splits", "-s", nargs="+", default=SPLITS, help="要统计的 split 列表")
    parser.add_argument("--out-json", default="cwe_counts.json", help="保存统计结果的 JSON 文件")
    parser.add_argument("--out-png", default="cwe_vulnerable_distribution_all.png", help="保存柱状图的文件名")
    parser.add_argument("--no-plot", action="store_true", help="只输出 JSON，不绘制图像")
    args = parser.parse_args()

    counts, total_checked, total_vuln = compute_vulnerable_cwe_counts(dataset_id=args.dataset, splits=args.splits)
    print(f"Total examples checked: {total_checked}, total vulnerable (is_vulnerable=True): {total_vuln}")
    print(f"Distinct CWE (incl UNKNOWN): {len(counts)}")
    # 按数量降序保存
    sorted_counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "splits": args.splits,
            "total_checked": total_checked,
            "total_vulnerable": total_vuln,
            "counts": sorted_counts
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved counts JSON to {args.out_json}")
    # 打印前20
    top20 = list(sorted_counts.items())[:20]
    print("Top 20 CWE:")
    for k, v in top20:
        print(f"{k}: {v}")
    if not args.no_plot:
        try:
            plot_all_counts(counts, out_png=args.out_png)
        except Exception as e:
            print(f"绘图失败: {e}")

if __name__ == "__main__":
    main()