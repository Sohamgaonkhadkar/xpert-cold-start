"""Aggregate official scores and error counts without exporting conversation text.

Usage: python -m scripts.evaluate_official INPUT OUTPUT [--report metrics/report.json]
Answers are used here only, never by the inference pipeline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import check


def evaluate(folder, prediction):
    chats, messages, seed, answers = check.load_input(folder)
    out = json.loads(Path(prediction).read_text(encoding="utf-8"))
    errors = check.validate(out, chats, messages)
    result = {"messages": len(messages), "chats": len(chats), "validation_errors": errors}
    if errors or answers is None:
        return result
    scores = check.score(out, answers, seed, messages)
    result.update(scores=scores, points={k: v * check.POINTS[k] if v is not None else None for k, v in scores.items()})
    result["total"] = sum(v for v in result["points"].values() if v is not None)
    result["possible"] = sum(check.POINTS[k] for k, v in scores.items() if v is not None)
    ts = {m["id"]: m["ts"] for m in messages}
    since = answers.get("loops_scored_from")
    predicted = [l for l in out["open_loops"] if since is None or ts[l["source_msg"]] >= since]
    actual = [l for l in answers["open_loops"] if since is None or min(ts[x] for x in l["source_msgs"]) >= since]
    used, pairs = set(), []
    for loop in predicted:
        for i, ref in enumerate(actual):
            if i not in used and loop["source_msg"] in ref["source_msgs"]:
                used.add(i)
                pairs.append((loop, ref))
                break
    hit = len(pairs)
    precision = hit / len(predicted) if predicted else 0.0
    recall = hit / len(actual) if actual else 0.0
    result["loops"] = {
        "predicted": len(predicted), "actual": len(actual), "matched": hit,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_done": sum(p["status"] == "done" and a["status"] != "done" for p, a in pairs),
        "missed_completions_matched": sum(p["status"] != "done" and a["status"] == "done" for p, a in pairs),
        "missed_completions_including_undiscovered": sum(a["status"] == "done" for a in actual) - sum(p["status"] == a["status"] == "done" for p, a in pairs),
        "false_dropped": sum(p["status"] == "dropped" and a["status"] != "dropped" for p, a in pairs),
        "incorrect_owner_matched": sum(p["owner"] != a.get("owner") for p, a in pairs if a.get("owner") is not None),
        "incorrect_intervention_on_leave": sum(a.get("help") == "leave" and (p.get("help") or {}).get("action") in ("assist", "remind") for p, a in pairs),
        "interventions_labelled": sum(bool(a.get("help")) for a in actual),
    }
    seeded = set((seed.get("circles") or {}).keys())
    circles = {p["id"]: p["circle"] for p in out["people"]}
    result["circle_confusions"] = dict(Counter(f'{p["circle"]} -> {circles.get(p["id"], "missing")}' for p in answers["people"] if p["id"] not in seeded and circles.get(p["id"]) != p["circle"]))
    result["goals_labelled"] = bool(answers.get("goals"))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = evaluate(args.input, args.output)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        dest = Path(args.report)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")
    print(rendered)
    return int(bool(result["validation_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
