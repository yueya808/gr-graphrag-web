import argparse
import csv
from pathlib import Path


def to_int(v):
    try:
        return int(float(str(v or "0")))
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Prefill manual score CSV with heuristic labels.")
    parser.add_argument("--input", required=True, help="manual_scores_template.csv path")
    parser.add_argument("--output", default="", help="output path (default: overwrite input)")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve() if args.output else in_path
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    with in_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else []

    for r in rows:
        method = (r.get("method") or "").strip().lower()
        category = (r.get("category") or "").strip()
        ans = str(r.get("answer") or "")
        kb = to_int(r.get("kb_count"))
        kg = to_int(r.get("kg_count"))
        kg_connected = str(r.get("kg_connected") or "").lower() in {"1", "true", "yes"}

        # is_correct
        if "无关问题" in category:
            is_correct = 1 if ("天气" in ans or "不相关" in ans or "无法" in ans) else 0
        else:
            if method == "graph_rag":
                is_correct = 1 if (kg > 0 or kg_connected) and kb > 0 else 0
            elif method == "rag_only":
                is_correct = 1 if kb > 0 else 0
            else:
                is_correct = 0

        # evidence_hit
        if method == "graph_rag":
            evidence_hit = 1 if (kg > 0 or kg_connected) else 0
        elif method == "rag_only":
            evidence_hit = 1 if kb > 0 else 0
        else:
            evidence_hit = 0

        # hallucination (1 = has hallucination)
        if "无关问题" in category:
            hallucination = 0 if ("无法" in ans or "不相关" in ans or "仅能回答" in ans) else 1
        else:
            if method == "graph_rag":
                hallucination = 0 if evidence_hit == 1 else 1
            elif method == "rag_only":
                hallucination = 0 if evidence_hit == 1 else 1
            else:
                hallucination = 1

        # root_path_hit
        if "无关问题" in category:
            root_path_hit = 0
        else:
            if method == "graph_rag":
                root_path_hit = 1 if ("原因" in ans and ("处理" in ans or "措施" in ans)) else 0
            elif method == "rag_only":
                root_path_hit = 1 if ("原因" in ans and ("建议" in ans or "处理" in ans)) else 0
            else:
                root_path_hit = 0

        r["is_correct"] = str(is_correct)
        r["evidence_hit"] = str(evidence_hit)
        r["hallucination"] = str(hallucination)
        r["root_path_hit"] = str(root_path_hit)
        if not (r.get("annotator") or "").strip():
            r["annotator"] = "auto_prefill"
        note = (r.get("notes") or "").strip()
        r["notes"] = (note + " | " if note else "") + "auto_prefill"

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)

    print(f"prefilled: {out_path}")
    print("请人工复核后再用于最终结果。")


if __name__ == "__main__":
    main()

