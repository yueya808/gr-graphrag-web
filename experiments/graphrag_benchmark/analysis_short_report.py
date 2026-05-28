import argparse
import csv
from collections import defaultdict
from pathlib import Path


def to_int01(value: str) -> int:
    v = str(value or "").strip()
    if v in {"1", "1.0", "true", "True", "TRUE", "是"}:
        return 1
    return 0


def pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def short_analyses(rows):
    lines = []
    rows_sorted = sorted(rows, key=lambda x: x["回答准确率"], reverse=True)
    best = rows_sorted[0]["方法"] if rows_sorted else "N/A"
    lines.append(f"准确率最高：{best}")
    low_h = sorted(rows, key=lambda x: x["幻觉率"])[0]["方法"] if rows else "N/A"
    lines.append(f"幻觉率最低：{low_h}")
    high_e = sorted(rows, key=lambda x: x["证据命中率"], reverse=True)[0]["方法"] if rows else "N/A"
    lines.append(f"证据最充分：{high_e}")
    high_r = sorted(rows, key=lambda x: x["根因路径命中率"], reverse=True)[0]["方法"] if rows else "N/A"
    lines.append(f"路径能力最强：{high_r}")
    fast = sorted(rows, key=lambda x: x["平均响应时间ms"])[0]["方法"] if rows else "N/A"
    lines.append(f"响应最快：{fast}")
    return [x[:20] for x in lines]


def main():
    parser = argparse.ArgumentParser(description="Generate summary table and short analysis.")
    parser.add_argument("--input", required=True, help="manual score csv path")
    parser.add_argument("--out-csv", default="", help="summary csv output path")
    args = parser.parse_args()

    path = Path(args.input).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else path.parent / "outputs" / "summary_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        data = list(csv.DictReader(f))

    by_method = defaultdict(list)
    for r in data:
        m = (r.get("method") or "").strip()
        if m:
            by_method[m].append(r)

    rows = []
    for method, items in by_method.items():
        n = len(items)
        acc = sum(to_int01(i.get("is_correct")) for i in items)
        hit = sum(to_int01(i.get("evidence_hit")) for i in items)
        hal = sum(to_int01(i.get("hallucination")) for i in items)
        root = sum(to_int01(i.get("root_path_hit")) for i in items)
        lat = []
        for i in items:
            try:
                lat.append(float(i.get("latency_ms") or 0))
            except Exception:
                pass
        avg_lat = round(sum(lat) / len(lat), 2) if lat else 0.0
        rows.append(
            {
                "方法": method,
                "回答准确率": pct(acc, n),
                "证据命中率": pct(hit, n),
                "幻觉率": pct(hal, n),
                "根因路径命中率": pct(root, n),
                "平均响应时间ms": avg_lat,
                "样本数": n,
            }
        )

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["方法", "回答准确率", "证据命中率", "幻觉率", "根因路径命中率", "平均响应时间ms", "样本数"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary saved: {out_csv}")
    print("方法,回答准确率,证据命中率,幻觉率,根因路径命中率,平均响应时间ms,样本数")
    for r in rows:
        print(
            f"{r['方法']},{r['回答准确率']}%,{r['证据命中率']}%,{r['幻觉率']}%,"
            f"{r['根因路径命中率']}%,{r['平均响应时间ms']},{r['样本数']}"
        )

    print("\n20字内分析：")
    for i, s in enumerate(short_analyses(rows), start=1):
        print(f"{i}. {s}")


if __name__ == "__main__":
    main()

