import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def to_int01(value: str) -> int:
    v = str(value or "").strip()
    if v in {"1", "1.0", "true", "True", "TRUE", "是"}:
        return 1
    return 0


def safe_pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(100.0 * num / den, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate manual evaluation scores.")
    parser.add_argument("--input", required=True, help="Path to manual score csv")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    rows: List[Dict[str, str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    by_method = defaultdict(list)
    for row in rows:
        method = (row.get("method") or "").strip()
        if method:
            by_method[method].append(row)

    print("方法,回答准确率,证据命中率,幻觉率,根因路径命中率,平均响应时间(ms),样本数")
    for method, items in by_method.items():
        total = len(items)
        correct = sum(to_int01(it.get("is_correct", "")) for it in items)
        hit = sum(to_int01(it.get("evidence_hit", "")) for it in items)
        halluc = sum(to_int01(it.get("hallucination", "")) for it in items)
        root_hit = sum(to_int01(it.get("root_path_hit", "")) for it in items)

        latency_vals = []
        for it in items:
            try:
                latency_vals.append(int(float(str(it.get("latency_ms", "0") or "0"))))
            except Exception:
                pass
        avg_latency = round(sum(latency_vals) / len(latency_vals), 2) if latency_vals else 0.0

        print(
            f"{method},{safe_pct(correct,total)}%,{safe_pct(hit,total)}%,{safe_pct(halluc,total)}%,"
            f"{safe_pct(root_hit,total)}%,{avg_latency},{total}"
        )


if __name__ == "__main__":
    main()
