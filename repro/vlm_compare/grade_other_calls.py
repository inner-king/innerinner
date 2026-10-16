# SPDX-License-Identifier: Apache-2.0
"""Side-by-side table of replay_other_calls.py results: masses and frictions (A11), front picks (A8),
cousin components and variation counts (B1), and how many proposed tasks parsed (B7).

There is no ground truth for these; the table is for reading next to the images (the README notes
which answers were judged right, and how).

Usage: python repro/vlm_compare/grade_other_calls.py <out.md> <label>=<other_calls.jsonl> [...]
"""
import json
import sys
from collections import defaultdict


def short(phrase):
    return (phrase or "").split(" / ")[0]


def main():
    out_md = sys.argv[1]
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    rows = defaultdict(dict)  # case id -> label -> cell
    meta = {}
    for label, path in runs:
        for line in open(path):
            r = json.loads(line)
            cid = r["case"]["id"]
            meta[cid] = r["case"]
            a = r.get("attempts") or []
            cost = f"{r['wall_s']:.0f}s" + (f", MAX_TOKENS {sum(x.get('done_reason') == 'length' for x in a)}"
                                             if any(x.get("done_reason") == "length" for x in a) else "")
            if r.get("other_gpu_jobs"):
                cost += ", contended"
            if r.get("error"):
                cell = f"ERROR {r['error'][:50]}"
            elif r["case"]["stage"] == "A11":
                cell = f"{r['mass'] * 1000:.0f} g, μ {r['friction']}"
            elif r["case"]["stage"] == "A8":
                cell = f"{r['choice'] or 'NONE'} ({r['front_azimuth_deg']}°)" if r.get("choice") else "NONE"
            elif cid.endswith("_components"):
                cell = ", ".join(r.get("components") or []) or "(empty)"
            elif cid.endswith("_variations"):
                cell = "; ".join(f"{d} {len(v)}" for dims in (r.get("variations") or {}).values() for d, v in dims.items()) or "(none)"
            elif cid.endswith("_topology"):
                cell = "yes" if r.get("topology") else "no"
            elif cid.endswith("_name"):
                cell = repr((r.get("text") or "").strip())
            else:  # B7
                cell = f"{r.get('n_tasks')}: " + ", ".join(c.get("task_name") or "?" for c in r.get("checks", []))
            rows[cid][label] = f"{cell} ({cost})"
    labels = [l for l, _ in runs]
    lines = ["| call | object | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for cid, cells in rows.items():
        obj = short(meta[cid].get("object")) + (f" / {meta[cid]['component']}" if meta[cid].get("component") else "")
        lines.append(f"| {cid} | {obj} | " + " | ".join(cells.get(l, "") for l in labels) + " |")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
