"""Build an explicitly synthetic 20k-message /320-chat runtime fixture.

This has sparse actionable tasks and cannot establish dense hidden-set quality.
Always write it to ignored scratch storage. No official conversations are used.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

from scripts.run_experiments import build_hard_8week_stress_benchmark


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    folder = build_hard_8week_stress_benchmark(Path(args.output_dir))
    chats = [json.loads(x) for x in (folder / "chats.jsonl").read_text(encoding="utf-8").splitlines()]
    messages = [json.loads(x) for x in (folder / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
    while len(chats) < 320:
        n = len(chats)
        chats.append({"chat_id": f"scale_chat_{n}", "name": f"Contact {n}", "is_group": False,
                      "participant_ids": ["ME", f"scale_person_{n}"]})
    start = datetime(2026, 9, 8, 8)
    templates = ["Good morning", "Traffic is heavy today.", "Ok noted", "Thanks", "[photo]"]
    while len(messages) < 20000:
        n = len(messages)
        chat = chats[n % len(chats)]
        sender = chat["participant_ids"][n % len(chat["participant_ids"])]
        messages.append({"id": f"scale_msg_{n}", "chat_id": chat["chat_id"], "sender_id": sender,
                         "sender_name": sender, "ts": (start + timedelta(seconds=n * 54 * 86400 / 20000)).isoformat(),
                         "kind": "text", "body": templates[n % len(templates)], "quoted_id": None})
    for chat in chats:
        # The production checker accepts the official key, not fixture aliases.
        chat["participants"] = chat.pop("participant_ids", chat.get("participants", []))
    for name, rows in (("chats", chats), ("messages", messages)):
        (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Synthetic runtime fixture: {len(messages)} messages /{len(chats)} chats")


if __name__ == "__main__":
    main()
