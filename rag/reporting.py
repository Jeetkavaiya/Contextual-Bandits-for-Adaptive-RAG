from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def save_metrics_summary_json(
    metrics_csv_path: str,
    out_json_path: str,
) -> None:
    """
    Reads metrics CSV produced by rag/evaluate.py and saves a compact JSON summary.

    Output JSON includes:
    - overall averages
    - in_domain averages
    - out_of_domain averages
    """
    import csv

    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(metrics_csv_path)

    rows: List[Dict[str, Any]] = []
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))

    def to_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none":
            return None
        try:
            return float(s)
        except Exception:
            return None

    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def summarize(domain: Optional[str]) -> Dict[str, Any]:
        rr = rows if domain is None else [r for r in rows if r.get("domain") == domain]

        corr = [to_float(r.get("correctness")) for r in rr]
        faith = [to_float(r.get("faithfulness")) for r in rr]
        recall = [to_float(r.get("evidence_recall_at_k")) for r in rr]
        t = [to_float(r.get("total_time_s")) for r in rr]
        tok = [to_float(r.get("total_tokens")) for r in rr]

        corr_f = [x for x in corr if x is not None]
        faith_f = [x for x in faith if x is not None]
        recall_f = [x for x in recall if x is not None]
        t_f = [x for x in t if x is not None]
        tok_f = [x for x in tok if x is not None]

        out = {
            "n": len(rr),
            "avg_correctness_0_2": avg(corr_f),
            "avg_faithfulness_0_1": avg(faith_f),
            "avg_total_time_s": avg(t_f),
            "avg_total_tokens": avg(tok_f),
        }
        if recall_f:
            out["avg_evidence_recall_at_k_0_1"] = avg(recall_f)
        return out

    summary = {
        "all": summarize(None),
        "in_domain": summarize("in_domain"),
        "out_of_domain": summarize("out_of_domain"),
        "source_metrics_csv": metrics_csv_path,
    }

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON: {out_json_path}")