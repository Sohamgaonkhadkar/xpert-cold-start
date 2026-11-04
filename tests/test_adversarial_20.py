"""Automated tests verifying all 20 required adversarial failure modes end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_synthetic_suite import build_synthetic_benchmark
from src.loop_extraction import resolve_due_date
from src.models import USER_ID
from src.output_builder import run_pipeline


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory: pytest.TempPathFactory):
    data_dir = tmp_path_factory.mktemp("synth_20")
    build_synthetic_benchmark(data_dir)
    out_obj, out_json = run_pipeline(data_dir)
    return data_dir, out_obj, out_json


def test_case_01_bare_acknowledgement_not_closed(synth_run):
    """#1: Bare 'Ok' / '👍' reply to proposal request must NOT mark loop done."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m101")
    assert loop.status == "open"
    assert loop.closure_message_id is None


def test_case_02_cross_chat_closure_propagation(synth_run):
    """#2: Client group request delegated in 1:1 DM and closed back in group marks loop done."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id in ("m110", "m111"))
    assert loop.status == "done"
    assert loop.closure_message_id in ("m112", "m113")


def test_case_03_delegated_task_updates_owner_and_leaves(synth_run):
    """#3: Task taken over by colleague ('Leave it to me, I'll handle this') reassigns owner and leaves."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m120")
    assert loop.owner == "p_colleague_2"
    assert loop.status == "open"
    assert loop.intervention is not None
    assert loop.intervention.action == "leave"
    assert loop.intervention.leave_reason == "someone_elses"


def test_case_04_polite_deferral_remains_open(synth_run):
    """#4: 'Haan kal subah pakka kar dunga, let me check' is a deferral, NOT completion."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m106")
    assert loop.status == "open"
    assert loop.closure_message_id is None


def test_case_05_sarcasm_and_complaint_not_closed(synth_run):
    """#5: 'Still not done from my side, bank cheque book mila nahi' must NOT trigger 'done'."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m135")
    assert loop.status == "open"
    assert loop.closure_message_id is None


def test_case_06_distinct_recurring_utility_tasks_not_collapsed(synth_run):
    """#6: Electricity bill (3 Nov) and Broadband bill (4 Nov) in same chat stay separate loops."""
    _, out, _ = synth_run
    m132 = [l for l in out.open_loops if l.message_id == "m132"]
    m133 = [l for l in out.open_loops if l.message_id == "m133"]
    assert len(m132) == 1 and m132[0].due_date == "2026-11-03"
    assert len(m133) == 1 and m133[0].due_date == "2026-11-04"


def test_case_07_rescheduled_deadline_updates_due_date(synth_run):
    """#7: Postponing Oct 26 lease renewal ('move the deadline to 5 Nov') updates due_date without duplicating."""
    _, out, _ = synth_run
    m125 = next(l for l in out.open_loops if l.message_id == "m125")
    assert m125.due_date == "2026-11-05"
    assert m125.status == "open"
    assert not any(l.message_id == "m126" for l in out.open_loops)


def test_case_08_cancelled_task_marked_dropped(synth_run):
    """#8: 'Rehne do mat aao, switch theek ho gaya, no longer needed' marks loop dropped."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m128")
    assert loop.status == "dropped"
    assert loop.closure_message_id == "m129"
    assert loop.intervention is None


def test_case_09_bilingual_hindi_request_english_closure(synth_run):
    """#9: Hindi salary request ('pagar bhej dena') closed by English reply ('Paid your salary via UPI')."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m001")
    assert loop.status == "done"
    assert loop.closure_message_id == "m002"


def test_case_10_bengali_transliterated_commitment_and_closure(synth_run):
    """#10: Romanized Bengali caretaker request ('beton ta pathiye dio') closed by 'UPI kore diyechi'."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m010")
    assert loop.status == "done"
    assert loop.closure_message_id == "m011"


def test_case_11_expired_deadline_never_auto_completed(synth_run):
    """#11: Consent form due 2026-11-02 (before dataset anchor 2026-11-03) with no reply stays OPEN."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m130")
    assert loop.due_date == "2026-11-02"
    assert loop.status == "open"
    assert loop.intervention is not None and loop.intervention.action == "remind"


def test_case_12_seed_conflict_anchored_to_seed_json(synth_run):
    """#12: Seeded person p_friend_1 saying 'bro' / cousin slang retains exact seed.json circle."""
    _, out, _ = synth_run
    p = next(p for p in out.people if p.id == "p_friend_1")
    assert p.circle == "friend"


def test_case_13_group_homophily_and_client_org_traps(synth_run):
    """#13: School parents group -> child person; Northstar client group -> client person (NOT user org)."""
    _, out, _ = synth_run
    org_names = {o.name.lower() for o in out.organisations}
    assert not any("northstar" in n for n in org_names)
    assert not any("class 4b" in n for n in org_names)

    chats_by_id = {c.chat_id: c for c in out.chats}
    assert chats_by_id["c_grp_class4b"].owner_type == "person"
    assert chats_by_id["c_grp_class4b"].owner_id == "p_child"
    assert chats_by_id["c_grp_northstar"].owner_type == "person"
    assert chats_by_id["c_grp_northstar"].owner_id == "p_client_1"


def test_case_14_duplicate_first_name_disambiguation(synth_run):
    """#14: 'Rahul Verma - Zephyr' (colleague) vs 'Rahul Electrician' (vendor) never bleed circles or closures."""
    _, out, _ = synth_run
    by_id = {p.id: p for p in out.people}
    assert by_id["p_colleague_1"].circle == "colleague"
    assert by_id["p_vendor_elec"].circle == "vendor"


def test_case_15_quoted_reply_yes_closes_completion_check(synth_run):
    """#15: 'Yes filed just now' quoting 'Did you submit the October GST return?' closes m103."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m103")
    assert loop.status == "done"
    assert loop.closure_message_id == "m105"


def test_case_16_noisy_voice_transcript_loop_and_assist(synth_run):
    """#16: Voice transcript request for Coorg/Goa beachfront resort shortlist extracted as open assist."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m139")
    assert loop.status == "open"
    assert loop.owner == USER_ID
    assert loop.intervention is not None
    assert loop.intervention.action == "assist"
    assert loop.intervention.needs_approval is False


def test_case_17_third_party_group_loop_marked_someone_elses(synth_run):
    """#17: Colleague-to-colleague task in group ('Rahul, send staging logs to Sneha') -> leave(someone_elses)."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m137")
    assert loop.owner == "p_colleague_1"
    assert loop.intervention is not None
    assert loop.intervention.action == "leave"
    assert loop.intervention.leave_reason == "someone_elses"


def test_case_18_sensitive_personal_loop_marked_leave_personal(synth_run):
    """#18: Intimate apology / anniversary note to spouse -> leave(personal), never ghostwritten."""
    _, out, _ = synth_run
    loop = next(l for l in out.open_loops if l.message_id == "m142")
    assert loop.status == "open"
    assert loop.intervention is not None
    assert loop.intervention.action == "leave"
    assert loop.intervention.leave_reason == "personal"


def test_case_19_repeated_followup_marked_leave_annoying(synth_run):
    """#19: Repeated nagging followups ('following up for the 3rd time... stop pinging') -> leave(annoying), and relative dates anchor to msg.ts."""
    from datetime import datetime

    from src.conversation_index import ConversationIndex
    from src.data_loader import load_dataset
    from src.intervention_policy import apply_intervention_policy
    from src.llm_client import LLMClient
    from src.models import EvidenceLedger, MessageRecord, OpenLoopPrediction

    data_dir, out, _ = synth_run
    bundle = load_dataset(data_dir)
    # Inject a nagging repeated-ping message into an active DM
    bundle.messages.append(
        MessageRecord(
            id="m_nag_99",
            chat_id="c_dm_friend2",
            sender_id="ME",
            sender_name="ME",
            ts=datetime(2026, 10, 31, 10, 0, 0),
            ts_raw="2026-10-31T10:00:00",
            kind="text",
            body="Following up for the 3rd time on RSVP, please stop pinging after this.",
            quoted_id=None,
        )
    )
    idx = ConversationIndex(bundle)
    people_map = {p.id: p for p in out.people}
    nag_loop = OpenLoopPrediction(
        id="loop_nag",
        title="Follow up on RSVP",
        type="followup",
        owner="ME",
        chat_id="c_dm_friend2",
        message_id="m_nag_99",
        status="open",
    )
    apply_intervention_policy([nag_loop], idx, people_map, LLMClient(api_key="", base_url=""), EvidenceLedger())
    assert nag_loop.intervention is not None
    assert nag_loop.intervention.action == "leave"
    assert nag_loop.intervention.leave_reason == "annoying"
    assert nag_loop.intervention.confidence >= 0.8

    msg_ts = datetime(2026, 10, 26, 9, 0, 0)
    assert resolve_due_date("Please send tomorrow", msg_ts) == "2026-10-27"
    assert resolve_due_date("Kal bheja tha", msg_ts) is None


def test_case_20_goals_precision_excludes_wishes_and_friend_goals(synth_run):
    """#20: Wishful thinking ('I wish I lived in Switzerland') and friend's marathon goal are excluded."""
    _, out, _ = synth_run
    titles = " | ".join(g.title.lower() for g in out.goals)
    assert "switzerland" not in titles
    assert "sub-4 marathon" not in titles
    assert 1 <= len(out.goals) <= 10
    # Staff care-support links are accurately inferred for Kabir and Maa
    driver = next(p for p in out.people if p.id == "p_staff_driver")
    assert set(driver.supports_people) == {"p_child", "p_parent_mom"}
