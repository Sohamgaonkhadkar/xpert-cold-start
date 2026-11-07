"""Offline, executable correction interview demonstration on the public example."""
import copy
from pathlib import Path

from src.conversation_index import ConversationIndex
from src.corrections import apply_corrections
from src.data_loader import load_dataset
from src.llm_client import LLMClient
from src.output_builder import run_pipeline


def main():
    data = Path(__file__).resolve().parent.parent / "example"
    # Explicit empty endpoint prevents accidental API use even in a configured shell.
    output, _ = run_pipeline(data, llm_client=LLMClient(base_url="", api_key=""))
    index = ConversationIndex(load_dataset(data))
    target = next(p for p in output.people if p.circle == "vendor")
    unrelated = {p.id: copy.deepcopy(p) for p in output.people if p.id != target.id}
    goals = copy.deepcopy(output.goals)
    notes = apply_corrections(output, {"people": {target.id: {"circle": "client"}}}, index)
    assert target.circle == "client" and output.goals == goals
    assert all(p == unrelated[p.id] for p in output.people if p.id in unrelated)
    loop = next(l for l in output.open_loops if l.status == "open")
    others = {l.id: copy.deepcopy(l) for l in output.open_loops if l.id != loop.id}
    apply_corrections(output, {"open_loops": {loop.message_id: {"owner": target.id}}}, index)
    assert loop.owner == target.id
    assert all(l == others[l.id] for l in output.open_loops if l.id in others)
    # Demonstrate a deliberate user correction with an actual later public message
    # as evidence; this is an interview exercise, not a scored prediction change.
    later = next(m for m in reversed(index.messages) if m.ts > index.messages_by_id[loop.message_id].ts)
    apply_corrections(output, {"open_loops": {loop.message_id: {
        "status": "done", "closure_message_id": later.id}}}, index)
    assert loop.status == "done" and loop.closure_message_id == later.id and loop.intervention is None
    assert all(l == others[l.id] for l in output.open_loops if l.id in others)
    print("PASS: relationship, owner and completion corrections; unrelated records stable")
    print(f"Public example targets: person={target.id}, source={loop.message_id}, evidence={later.id}")


if __name__ == "__main__":
    main()
