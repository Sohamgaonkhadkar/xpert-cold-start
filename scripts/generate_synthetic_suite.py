"""Synthetic Multilingual & 20-Case Adversarial Benchmark Generator for Xpert Cold Start."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_synthetic_benchmark(dest_dir: str | Path) -> Path:
    """Write a complete 2-week multilingual dataset + seed.json + answers/reference.json covering all 12 circles, 4 orgs, 5 goals, and all 20 adversarial cases."""
    out = Path(dest_dir)
    (out / "answers").mkdir(parents=True, exist_ok=True)

    # 22 People covering all 12 circles + adversarial confusers (#13 Didi staff vs Didi sibling, #14 two Rahuls)
    people_truth = [
        {"id": "p_spouse", "name": "Neha", "circle": "spouse", "organisation_links": [], "supports_people": []},
        {"id": "p_child", "name": "Kabir", "circle": "child", "organisation_links": [], "supports_people": []},
        {"id": "p_parent_mom", "name": "Maa", "circle": "parent", "organisation_links": [], "supports_people": []},
        {"id": "p_inlaw_dad", "name": "Papaji", "circle": "in-law", "organisation_links": [], "supports_people": []},
        {"id": "p_sibling", "name": "Ananya Didi", "circle": "sibling", "organisation_links": [], "supports_people": []},
        {"id": "p_relative", "name": "Suresh Mama", "circle": "relative", "organisation_links": [], "supports_people": []},
        {"id": "p_friend_1", "name": "Dev", "circle": "friend", "organisation_links": [], "supports_people": []},
        {"id": "p_friend_2", "name": "Ritam", "circle": "friend", "organisation_links": [], "supports_people": []},
        {"id": "p_partner", "name": "Vikram - Zephyr", "circle": "partner", "organisation_links": [{"organisation_id": "org_zephyr", "relation": "runs"}], "supports_people": []},
        {"id": "p_colleague_1", "name": "Rahul Verma - Zephyr", "circle": "colleague", "organisation_links": [{"organisation_id": "org_zephyr", "relation": "runs"}], "supports_people": []},
        {"id": "p_colleague_2", "name": "Sneha - Zephyr", "circle": "colleague", "organisation_links": [{"organisation_id": "org_zephyr", "relation": "runs"}], "supports_people": []},
        {"id": "p_colleague_3", "name": "Ishaan - Kite", "circle": "colleague", "organisation_links": [{"organisation_id": "org_kite", "relation": "runs"}], "supports_people": []},
        {"id": "p_client_1", "name": "tarun@northstar", "circle": "client", "organisation_links": [], "supports_people": []},
        {"id": "p_client_2", "name": "Kavita Menon", "circle": "client", "organisation_links": [], "supports_people": []},
        {"id": "p_vendor_ca", "name": "CA Manish Gupta", "circle": "vendor", "organisation_links": [], "supports_people": []},
        {"id": "p_vendor_elec", "name": "Rahul Electrician", "circle": "vendor", "organisation_links": [{"organisation_id": "org_home", "relation": "supports"}], "supports_people": []},
        {"id": "p_staff_driver", "name": "Manoj Driver", "circle": "staff", "organisation_links": [{"organisation_id": "org_home", "relation": "supports"}], "supports_people": ["p_child", "p_parent_mom"]},
        {"id": "p_staff_cook", "name": "Lata Didi", "circle": "staff", "organisation_links": [{"organisation_id": "org_home", "relation": "supports"}], "supports_people": []},
        {"id": "p_staff_villa", "name": "Bapi Caretaker", "circle": "staff", "organisation_links": [{"organisation_id": "org_villa", "relation": "supports"}], "supports_people": []},
        {"id": "p_soc_sec", "name": "Col Bhatia Flat 802", "circle": "friend", "organisation_links": [], "supports_people": []},
    ]

    # Seed (~20% = 4 people, 2 org names, 1 goal)
    seed_data: Dict[str, Any] = {
        "people": {
            "p_spouse": "spouse",
            "p_child": "child",
            "p_partner": "partner",
            "p_friend_1": "friend",
        },
        "organisations": [
            "Zephyr Labs",
            "Kite Studio",
        ],
        "goals": [
            {
                "title": "Run 3 times a week before 8am",
                "area": "health",
                "type": "habit",
                "horizon": "3mo",
            }
        ],
    }

    # Chats (1:1 DMs + Groups including child's class group #11, client project group, society group, villa group, home group, company groups)
    chats: List[Dict[str, Any]] = [
        {"chat_id": "c_dm_spouse", "name": "Neha", "is_group": False, "participant_ids": ["ME", "p_spouse"]},
        {"chat_id": "c_dm_child", "name": "Kabir", "is_group": False, "participant_ids": ["ME", "p_child"]},
        {"chat_id": "c_dm_mom", "name": "Maa", "is_group": False, "participant_ids": ["ME", "p_parent_mom"]},
        {"chat_id": "c_dm_inlaw", "name": "Papaji", "is_group": False, "participant_ids": ["ME", "p_inlaw_dad"]},
        {"chat_id": "c_dm_sibling", "name": "Ananya Didi", "is_group": False, "participant_ids": ["ME", "p_sibling"]},
        {"chat_id": "c_dm_relative", "name": "Suresh Mama", "is_group": False, "participant_ids": ["ME", "p_relative"]},
        {"chat_id": "c_dm_friend1", "name": "Dev", "is_group": False, "participant_ids": ["ME", "p_friend_1"]},
        {"chat_id": "c_dm_friend2", "name": "Ritam", "is_group": False, "participant_ids": ["ME", "p_friend_2"]},
        {"chat_id": "c_dm_partner", "name": "Vikram - Zephyr", "is_group": False, "participant_ids": ["ME", "p_partner"]},
        {"chat_id": "c_dm_col1", "name": "Rahul Verma - Zephyr", "is_group": False, "participant_ids": ["ME", "p_colleague_1"]},
        {"chat_id": "c_dm_col2", "name": "Sneha - Zephyr", "is_group": False, "participant_ids": ["ME", "p_colleague_2"]},
        {"chat_id": "c_dm_col3", "name": "Ishaan - Kite", "is_group": False, "participant_ids": ["ME", "p_colleague_3"]},
        {"chat_id": "c_dm_client1", "name": "tarun@northstar", "is_group": False, "participant_ids": ["ME", "p_client_1"]},
        {"chat_id": "c_dm_client2", "name": "Kavita Menon", "is_group": False, "participant_ids": ["ME", "p_client_2"]},
        {"chat_id": "c_dm_ca", "name": "CA Manish Gupta", "is_group": False, "participant_ids": ["ME", "p_vendor_ca"]},
        {"chat_id": "c_dm_elec", "name": "Rahul Electrician", "is_group": False, "participant_ids": ["ME", "p_vendor_elec"]},
        {"chat_id": "c_dm_driver", "name": "Manoj Driver", "is_group": False, "participant_ids": ["ME", "p_staff_driver"]},
        {"chat_id": "c_dm_cook", "name": "Lata Didi", "is_group": False, "participant_ids": ["ME", "p_staff_cook"]},
        {"chat_id": "c_dm_villa", "name": "Bapi Caretaker", "is_group": False, "participant_ids": ["ME", "p_staff_villa"]},
        # Group chats
        {"chat_id": "c_grp_zephyr_core", "name": "Zephyr Labs Core", "is_group": True, "participant_ids": ["ME", "p_partner", "p_colleague_1", "p_colleague_2"]},
        {"chat_id": "c_grp_kite_ops", "name": "Kite Studio Ops", "is_group": True, "participant_ids": ["ME", "p_colleague_3"]},
        {"chat_id": "c_grp_home", "name": "Home Ops & Kitchen", "is_group": True, "participant_ids": ["ME", "p_spouse", "p_staff_driver", "p_staff_cook", "p_vendor_elec"]},
        {"chat_id": "c_grp_society", "name": "Palm Grove Society RWA", "is_group": True, "participant_ids": ["ME", "p_spouse", "p_soc_sec"]},
        {"chat_id": "c_grp_villa", "name": "Shantiniketan Holiday Home", "is_group": True, "participant_ids": ["ME", "p_spouse", "p_staff_villa"]},
        {"chat_id": "c_grp_class4b", "name": "Class 4B Parents Kabir", "is_group": True, "participant_ids": ["ME", "p_spouse", "p_soc_sec"]},
        {"chat_id": "c_grp_northstar", "name": "Northstar x Zephyr Pilot", "is_group": True, "participant_ids": ["ME", "p_colleague_1", "p_client_1"]},
        {"chat_id": "c_grp_friends", "name": "Weekend Football Gang", "is_group": True, "participant_ids": ["ME", "p_friend_1", "p_friend_2"]},
    ]

    messages: List[Dict[str, Any]] = [
        # --- Relationship & Multilingual Cues (#12 Hinglish/Bengali, #13 Lata Didi cook vs Ananya Didi sibling, #14 two Rahuls) ---
        {"id": "m001", "chat_id": "c_dm_cook", "sender_id": "p_staff_cook", "sender_name": "Lata Didi", "ts": "2026-10-18T07:30:00", "kind": "voice_transcript", "body": "Sahab aaj sabzi aur nasta bana diya hai, kal mujhe ek din ki chutti chahiye aur is mahine ki pagar bhej dena.", "quoted_id": None},
        {"id": "m002", "chat_id": "c_dm_cook", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-18T08:05:00", "kind": "text", "body": "Paid your salary via UPI just now Lata Didi, chutti theek hai.", "quoted_id": "m001"},
        {"id": "m003", "chat_id": "c_dm_sibling", "sender_id": "p_sibling", "sender_name": "Ananya Didi", "ts": "2026-10-18T10:00:00", "kind": "text", "body": "Bhai, Maa ko kal doctor ke paas le jaana hai aur Rakhi ke photos dekhe? Papa bol rahe the hum dono milte hain.", "quoted_id": None},
        {"id": "m004", "chat_id": "c_dm_sibling", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-18T10:15:00", "kind": "text", "body": "Haan Didi, Maa ko main bol deta hoon, Manoj driver le jayega clinic.", "quoted_id": "m003"},
        {"id": "m005", "chat_id": "c_dm_mom", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-18T10:20:00", "kind": "text", "body": "Pranam Maa, please take your medicine on time. Manoj will pick up you for clinic checkup.", "quoted_id": None},
        {"id": "m006", "chat_id": "c_dm_mom", "sender_id": "p_parent_mom", "sender_name": "Maa", "ts": "2026-10-18T10:40:00", "kind": "text", "body": "Aashirwad beta, dawai le li hai.", "quoted_id": "m005"},
        {"id": "m007", "chat_id": "c_dm_inlaw", "sender_id": "p_inlaw_dad", "sender_name": "Papaji", "ts": "2026-10-19T09:00:00", "kind": "text", "body": "Beta Neha aur Mummyji bata rahe the Diwali par aap sab hamare ghar aaoge.", "quoted_id": None},
        {"id": "m008", "chat_id": "c_dm_relative", "sender_id": "p_relative", "sender_name": "Suresh Mama", "ts": "2026-10-19T11:00:00", "kind": "text", "body": "Beta Mami and your cousin are visiting Kolkata next month.", "quoted_id": None},
        {"id": "m009", "chat_id": "c_dm_friend2", "sender_id": "p_friend_2", "sender_name": "Ritam", "ts": "2026-10-19T19:00:00", "kind": "text", "body": "Yaar bondhu Sunday football match ke baad beer aur chill karte hain!", "quoted_id": None},
        {"id": "m010", "chat_id": "c_dm_villa", "sender_id": "p_staff_villa", "sender_name": "Bapi Caretaker", "ts": "2026-10-19T16:00:00", "kind": "text", "body": "Dada Shantiniketan bagan porishkar kore diyechi, gari gate e ache, beton ta pathiye dio.", "quoted_id": None},
        {"id": "m011", "chat_id": "c_dm_villa", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-19T16:30:00", "kind": "text", "body": "Pathiye diyechi Bapi, check kore nio.", "quoted_id": "m010"},
        {"id": "m012", "chat_id": "c_dm_driver", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-20T07:45:00", "kind": "text", "body": "Manoj, roz 2pm Kabir ko school se pick up karke tennis class chod dena aur Maa ko clinic le aana.", "quoted_id": None},
        {"id": "m013", "chat_id": "c_dm_driver", "sender_id": "p_staff_driver", "sender_name": "Manoj Driver", "ts": "2026-10-20T07:50:00", "kind": "text", "body": "Ji sahab, gaadi nikal li hai.", "quoted_id": "m012"},
        {"id": "m014", "chat_id": "c_dm_client2", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-20T11:30:00", "kind": "text", "body": "Hi Kavita, our team prepared the SOW and onboarding proposal for your pilot engagement.", "quoted_id": None},
        {"id": "m015", "chat_id": "c_dm_client2", "sender_id": "p_client_2", "sender_name": "Kavita Menon", "ts": "2026-10-20T12:00:00", "kind": "text", "body": "Thanks, reviewing internally with procurement.", "quoted_id": "m014"},
        {"id": "m016", "chat_id": "c_grp_zephyr_core", "sender_id": "p_colleague_2", "sender_name": "Sneha - Zephyr", "ts": "2026-10-20T14:00:00", "kind": "text", "body": "Team standup: staging deploy and sprint PR merge completed for release.", "quoted_id": None},
        {"id": "m017", "chat_id": "c_grp_kite_ops", "sender_id": "p_colleague_3", "sender_name": "Ishaan - Kite", "ts": "2026-10-20T15:00:00", "kind": "text", "body": "Updated the Kite Studio sprint roadmap and candidate interview pipeline.", "quoted_id": None},

        # --- Goals: 4 real goals + #10 two trap false goals (weak joke + friend's marathon) ---
        {"id": "m020", "chat_id": "c_dm_friend1", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-21T07:15:00", "kind": "text", "body": "Finished my 5k! Sticking to my goal to run 3 times a week before 8am.", "quoted_id": None},
        {"id": "m021", "chat_id": "c_grp_zephyr_core", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-21T10:00:00", "kind": "text", "body": "Our goal is to sign 12 paying enterprise clients by March so we hit profitability.", "quoted_id": None},
        {"id": "m022", "chat_id": "c_dm_spouse", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-21T20:00:00", "kind": "text", "body": "Planning our family vacation trip to Coorg in December, let's lock dates.", "quoted_id": None},
        {"id": "m023", "chat_id": "c_dm_child", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-22T18:00:00", "kind": "text", "body": "Beta my goal is to help you score 90+ in your Class 4 maths exam by December, let's practice fractions.", "quoted_id": None},
        # Trap #10a: Weakly implied joke wish by ME
        {"id": "m024", "chat_id": "c_grp_friends", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-22T21:00:00", "kind": "text", "body": "Traffic was brutal today, wish I lived in a cabin in Norway lol haha", "quoted_id": None},
        # Trap #10b: Friend's own goal, NOT ME's goal
        {"id": "m025", "chat_id": "c_grp_friends", "sender_id": "p_friend_2", "sender_name": "Ritam", "ts": "2026-10-22T21:05:00", "kind": "text", "body": "My goal is to run the full Tata Mumbai Marathon in January!", "quoted_id": None},
        {"id": "m026", "chat_id": "c_grp_friends", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-22T21:06:00", "kind": "text", "body": "Awesome bro, cheering for you!", "quoted_id": "m025"},

        # --- Open Loops: All 20 Adversarial Cases ---
        # Case #1 & #17: "Ok" meaning ACKNOWLEDGEMENT rather than completion (MUST stay OPEN; #20a assist needs_approval=True)
        {"id": "m101", "chat_id": "c_dm_client1", "sender_id": "p_client_1", "sender_name": "tarun@northstar", "ts": "2026-10-26T10:00:00", "kind": "text", "body": "Can you please send the revised Q4 security questionnaire proposal by tomorrow?", "quoted_id": None},
        {"id": "m102", "chat_id": "c_dm_client1", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-26T10:05:00", "kind": "text", "body": "Ok", "quoted_id": "m101"},

        # Case #2 & #17: "Ok" / "Haan" genuinely confirming completion in reply to a status-check question (MUST be DONE via m105)
        {"id": "m103", "chat_id": "c_dm_ca", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-25T11:00:00", "kind": "text", "body": "Manish ji, please file the October GST return before Wednesday.", "quoted_id": None},
        {"id": "m104", "chat_id": "c_dm_ca", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-27T17:00:00", "kind": "text", "body": "Did you submit the October GST return?", "quoted_id": "m103"},
        {"id": "m105", "chat_id": "c_dm_ca", "sender_id": "p_vendor_ca", "sender_name": "CA Manish Gupta", "ts": "2026-10-27T17:10:00", "kind": "text", "body": "Haanji done", "quoted_id": "m104"},

        # Case #3: Staff member taking responsibility (owner -> p_staff_driver, status -> open, intervention -> leave: someone_elses)
        {"id": "m106", "chat_id": "c_grp_home", "sender_id": "p_spouse", "sender_name": "Neha", "ts": "2026-10-29T09:00:00", "kind": "text", "body": "Manoj, please get the Honda car tyre pressure and servicing checked tomorrow.", "quoted_id": None},
        {"id": "m107", "chat_id": "c_grp_home", "sender_id": "p_staff_driver", "sender_name": "Manoj Driver", "ts": "2026-10-29T09:05:00", "kind": "text", "body": "Main dekh lunga bhabhiji.", "quoted_id": "m106"},

        # Case #4: Spouse discussing her own non-actionable office lunch (NO loop should be extracted!)
        {"id": "m108", "chat_id": "c_dm_spouse", "sender_id": "p_spouse", "sender_name": "Neha", "ts": "2026-10-28T13:00:00", "kind": "text", "body": "Having team lunch at my office cafeteria with Ritu today, great pasta.", "quoted_id": None},

        # Case #5 & #16: One task across multiple chats & phrased differently, closed in the group chat!
        {"id": "m110", "chat_id": "c_grp_northstar", "sender_id": "p_client_1", "sender_name": "tarun@northstar", "ts": "2026-10-26T14:00:00", "kind": "text", "body": "Please share the API latency benchmark deck before Thursday.", "quoted_id": None},
        {"id": "m111", "chat_id": "c_dm_col1", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-26T14:15:00", "kind": "text", "body": "Rahul, please send the API latency benchmark presentation for Northstar.", "quoted_id": None},
        {"id": "m112", "chat_id": "c_dm_col1", "sender_id": "p_colleague_1", "sender_name": "Rahul Verma - Zephyr", "ts": "2026-10-27T11:00:00", "kind": "text", "body": "Bhej diya latency benchmark slides on email and group!", "quoted_id": "m111"},
        {"id": "m113", "chat_id": "c_grp_northstar", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-27T11:05:00", "kind": "text", "body": "Shared the API latency benchmark deck over email just now.", "quoted_id": "m110"},

        # Case #6: Completed task (m115->m116 DONE) followed next day by a SEPARATE revision request (m117 OPEN)
        {"id": "m115", "chat_id": "c_dm_partner", "sender_id": "p_partner", "sender_name": "Vikram - Zephyr", "ts": "2026-10-25T15:00:00", "kind": "text", "body": "Can you review and send the Q3 investor update draft?", "quoted_id": None},
        {"id": "m116", "chat_id": "c_dm_partner", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-25T18:30:00", "kind": "text", "body": "Sent the Q3 investor update draft with edits.", "quoted_id": "m115"},
        {"id": "m117", "chat_id": "c_dm_partner", "sender_id": "p_partner", "sender_name": "Vikram - Zephyr", "ts": "2026-10-29T10:00:00", "kind": "text", "body": "Can you also prepare the November hiring budget spreadsheet by Saturday?", "quoted_id": None},

        # Case #7: Changing task ownership (starts as ME commitment m120, colleague Sneha takes over at m121 -> owner=p_colleague_2, leave!)
        {"id": "m120", "chat_id": "c_grp_zephyr_core", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-28T11:00:00", "kind": "text", "body": "I'll update the customer onboarding Notion docs by Friday.", "quoted_id": None},
        {"id": "m121", "chat_id": "c_grp_zephyr_core", "sender_id": "p_colleague_2", "sender_name": "Sneha - Zephyr", "ts": "2026-10-28T11:20:00", "kind": "text", "body": "Leave it to me, I'll handle this customer onboarding Notion docs update.", "quoted_id": "m120"},

        # Case #8: Postponed deadline (initially tomorrow Oct 28 -> postponed to 2026-11-05, stays OPEN!)
        {"id": "m125", "chat_id": "c_dm_col3", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-27T16:00:00", "kind": "text", "body": "I'll call the vendor and sign the studio lease renewal tomorrow.", "quoted_id": None},
        {"id": "m126", "chat_id": "c_dm_col3", "sender_id": "p_colleague_3", "sender_name": "Ishaan - Kite", "ts": "2026-10-28T09:30:00", "kind": "text", "body": "Landlord is travelling, let's postpone and move the deadline to 5 Nov.", "quoted_id": "m125"},

        # Case #9: Explicit cancellation (status = DROPPED, closure_message_id = m129)
        {"id": "m128", "chat_id": "c_dm_elec", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-26T09:00:00", "kind": "text", "body": "Rahul bhaiya, kal aake balcony light switch change kar dena.", "quoted_id": None},
        {"id": "m129", "chat_id": "c_dm_elec", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-26T18:00:00", "kind": "text", "body": "Rehne do mat aao, switch theek ho gaya, no longer needed.", "quoted_id": "m128"},

        # Case #11: Child's school class group deadline notice (chat -> p_child; loop -> deadline OPEN, remind ME)
        {"id": "m130", "chat_id": "c_grp_class4b", "sender_id": "p_spouse", "sender_name": "Neha", "ts": "2026-10-29T14:00:00", "kind": "image_caption", "body": "Arjun, school science excursion consent form physical signature due on 2 Nov, please sign tonight.", "quoted_id": None},

        # Case #15: Similar tasks that MUST NOT be merged (Electricity bill vs Broadband bill, both OPEN)
        {"id": "m132", "chat_id": "c_grp_home", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-29T19:00:00", "kind": "document", "body": "I'll pay the October electricity bill via OTP before 3 Nov.", "quoted_id": None},
        {"id": "m133", "chat_id": "c_grp_home", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-29T19:05:00", "kind": "document", "body": "I'll pay the October broadband bill via OTP before 4 Nov.", "quoted_id": None},

        # Case #18: Deadline that has PASSED without confirmation of completion (due 2026-10-24 < anchor 2026-10-30, MUST remain OPEN!)
        {"id": "m135", "chat_id": "c_grp_society", "sender_id": "p_soc_sec", "sender_name": "Col Bhatia Flat 802", "ts": "2026-10-21T10:00:00", "kind": "text", "body": "Arjun, society quarterly maintenance cheque submission deadline is 24 Oct, please hand over at office.", "quoted_id": None},

        # Case #19: Someone else's task between two non-ME people in a group (owner=p_colleague_1, leave: someone_elses)
        {"id": "m137", "chat_id": "c_grp_zephyr_core", "sender_id": "p_partner", "sender_name": "Vikram - Zephyr", "ts": "2026-10-30T11:00:00", "kind": "text", "body": "Rahul, please send the staging server AWS logs to Sneha by evening.", "quoted_id": None},

        # Case #20b: Task the assistant CAN prepare privately without outward side effects (assist, needs_approval = False!)
        {"id": "m139", "chat_id": "c_dm_spouse", "sender_id": "p_spouse", "sender_name": "Neha", "ts": "2026-10-30T16:00:00", "kind": "text", "body": "Can you find and shortlist two good beachfront resort options in Coorg or Goa for December?", "quoted_id": None},
        {"id": "m140", "chat_id": "c_dm_spouse", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-30T16:10:00", "kind": "text", "body": "I'll shortlist the resort options this weekend.", "quoted_id": "m139"},

        # Intimate personal spouse message -> leave: personal
        {"id": "m142", "chat_id": "c_dm_spouse", "sender_id": "ME", "sender_name": "ME", "ts": "2026-10-30T21:30:00", "kind": "text", "body": "Sorry I was stressed today, I'll write you a personal anniversary note.", "quoted_id": None},
    ]

    reference_output: Dict[str, Any] = {
        "people": people_truth,
        "organisations": [
            {
                "id": "org_zephyr",
                "name": "Zephyr Labs",
                "type": "company",
                "side": "work",
                "role": "partner",
                "people": ["p_colleague_1", "p_colleague_2", "p_partner"],
                "chats": ["c_grp_zephyr_core"],
            },
            {
                "id": "org_kite",
                "name": "Kite Studio",
                "type": "company",
                "side": "work",
                "role": "owner",
                "people": ["p_colleague_3"],
                "chats": ["c_grp_kite_ops"],
            },
            {
                "id": "org_home",
                "name": "Home",
                "type": "residence",
                "side": "home",
                "role": "owner",
                "people": ["p_staff_cook", "p_staff_driver", "p_vendor_elec"],
                "chats": ["c_grp_home"],
            },
            {
                "id": "org_society",
                "name": "Palm Grove Society RWA",
                "type": "society",
                "side": "home",
                "role": "resident",
                "people": [],
                "chats": ["c_grp_society"],
            },
            {
                "id": "org_villa",
                "name": "Shantiniketan Holiday Home",
                "type": "vacation home",
                "side": "home",
                "role": "owner",
                "people": ["p_staff_villa"],
                "chats": ["c_grp_villa"],
            },
        ],
        "chats": [
            {"chat_id": "c_dm_spouse", "owner_type": "person", "owner_id": "p_spouse"},
            {"chat_id": "c_dm_child", "owner_type": "person", "owner_id": "p_child"},
            {"chat_id": "c_dm_mom", "owner_type": "person", "owner_id": "p_parent_mom"},
            {"chat_id": "c_dm_inlaw", "owner_type": "person", "owner_id": "p_inlaw_dad"},
            {"chat_id": "c_dm_sibling", "owner_type": "person", "owner_id": "p_sibling"},
            {"chat_id": "c_dm_relative", "owner_type": "person", "owner_id": "p_relative"},
            {"chat_id": "c_dm_friend1", "owner_type": "person", "owner_id": "p_friend_1"},
            {"chat_id": "c_dm_friend2", "owner_type": "person", "owner_id": "p_friend_2"},
            {"chat_id": "c_dm_partner", "owner_type": "person", "owner_id": "p_partner"},
            {"chat_id": "c_dm_col1", "owner_type": "person", "owner_id": "p_colleague_1"},
            {"chat_id": "c_dm_col2", "owner_type": "person", "owner_id": "p_colleague_2"},
            {"chat_id": "c_dm_col3", "owner_type": "person", "owner_id": "p_colleague_3"},
            {"chat_id": "c_dm_client1", "owner_type": "person", "owner_id": "p_client_1"},
            {"chat_id": "c_dm_client2", "owner_type": "person", "owner_id": "p_client_2"},
            {"chat_id": "c_dm_ca", "owner_type": "person", "owner_id": "p_vendor_ca"},
            {"chat_id": "c_dm_elec", "owner_type": "person", "owner_id": "p_vendor_elec"},
            {"chat_id": "c_dm_driver", "owner_type": "person", "owner_id": "p_staff_driver"},
            {"chat_id": "c_dm_cook", "owner_type": "person", "owner_id": "p_staff_cook"},
            {"chat_id": "c_dm_villa", "owner_type": "person", "owner_id": "p_staff_villa"},
            {"chat_id": "c_grp_zephyr_core", "owner_type": "organisation", "owner_id": "org_zephyr"},
            {"chat_id": "c_grp_kite_ops", "owner_type": "organisation", "owner_id": "org_kite"},
            {"chat_id": "c_grp_home", "owner_type": "organisation", "owner_id": "org_home"},
            {"chat_id": "c_grp_society", "owner_type": "organisation", "owner_id": "org_society"},
            {"chat_id": "c_grp_villa", "owner_type": "organisation", "owner_id": "org_villa"},
            {"chat_id": "c_grp_class4b", "owner_type": "person", "owner_id": "p_child"},
            {"chat_id": "c_grp_northstar", "owner_type": "person", "owner_id": "p_client_1"},
            {"chat_id": "c_grp_friends", "owner_type": "person", "owner_id": "p_friend_2"},
        ],
        "goals": [
            {"id": "g1", "title": "Run 3 times a week before 8am", "area": "health", "type": "habit", "horizon": "3mo", "message_ids": ["m020"]},
            {"id": "g2", "title": "Sign 12 paying enterprise clients by March", "area": "work", "type": "milestone", "horizon": "3mo", "message_ids": ["m021"]},
            {"id": "g3", "title": "Family vacation trip to Coorg in December", "area": "experiences", "type": "experience", "horizon": "3mo", "message_ids": ["m022", "m139", "m140"]},
            {"id": "g4", "title": "Help Kabir score 90+ in Class 4 maths exam by December", "area": "kids", "type": "milestone", "horizon": "3mo", "message_ids": ["m023"]},
        ],
        "open_loops": [
            {
                "id": "rl_cook_salary",
                "title": "Send Lata Didi monthly salary",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_cook",
                "message_id": "m001",
                "acceptable_origin_ids": ["m001"],
                "due_date": "2026-10-19",
                "status": "done",
                "closure_message_id": "m002",
                "acceptable_closure_ids": ["m002"],
                "intervention": None,
            },
            {
                "id": "rl_villa_salary",
                "title": "Send Bapi caretaker salary",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_villa",
                "message_id": "m010",
                "acceptable_origin_ids": ["m010"],
                "due_date": None,
                "status": "done",
                "closure_message_id": "m011",
                "acceptable_closure_ids": ["m011"],
                "intervention": None,
            },
            {
                "id": "rl_driver_routine",
                "title": "Manoj pick up Kabir from school and Maa for clinic",
                "type": "followup",
                "owner": "p_staff_driver",
                "chat_id": "c_dm_driver",
                "message_id": "m012",
                "acceptable_origin_ids": ["m012"],
                "due_date": None,
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "leave", "leave_reason": "someone_elses"},
            },
            {
                "id": "rl_case1_ok_ack",
                "title": "Send revised Q4 security questionnaire proposal",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_client1",
                "message_id": "m101",
                "acceptable_origin_ids": ["m101", "m102"],
                "due_date": "2026-10-27",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "assist", "needs_approval": True},
            },
            {
                "id": "rl_case2_ok_done",
                "title": "File October GST return",
                "type": "followup",
                "owner": "p_vendor_ca",
                "chat_id": "c_dm_ca",
                "message_id": "m103",
                "acceptable_origin_ids": ["m103"],
                "due_date": "2026-10-28",
                "status": "done",
                "closure_message_id": "m105",
                "acceptable_closure_ids": ["m105"],
                "intervention": None,
            },
            {
                "id": "rl_case3_staff_car",
                "title": "Get Honda car tyre pressure and servicing checked",
                "type": "followup",
                "owner": "p_staff_driver",
                "chat_id": "c_grp_home",
                "message_id": "m106",
                "acceptable_origin_ids": ["m106", "m107"],
                "due_date": "2026-10-30",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "leave", "leave_reason": "someone_elses"},
            },
            {
                "id": "rl_case5_crosschat",
                "title": "Share API latency benchmark deck with Northstar",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_grp_northstar",
                "message_id": "m110",
                "acceptable_origin_ids": ["m110", "m111"],
                "due_date": "2026-10-29",
                "status": "done",
                "closure_message_id": "m113",
                "acceptable_closure_ids": ["m112", "m113"],
                "intervention": None,
            },
            {
                "id": "rl_case6_part1_done",
                "title": "Review and send Q3 investor update draft",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_partner",
                "message_id": "m115",
                "acceptable_origin_ids": ["m115"],
                "due_date": None,
                "status": "done",
                "closure_message_id": "m116",
                "acceptable_closure_ids": ["m116"],
                "intervention": None,
            },
            {
                "id": "rl_case6_part2_open",
                "title": "Prepare November hiring budget spreadsheet",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_partner",
                "message_id": "m117",
                "acceptable_origin_ids": ["m117"],
                "due_date": "2026-10-31",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "assist", "needs_approval": False},
            },
            {
                "id": "rl_case7_takeover",
                "title": "Update customer onboarding Notion docs",
                "type": "commitment",
                "owner": "p_colleague_2",
                "chat_id": "c_grp_zephyr_core",
                "message_id": "m120",
                "acceptable_origin_ids": ["m120", "m121"],
                "due_date": "2026-10-30",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "leave", "leave_reason": "someone_elses"},
            },
            {
                "id": "rl_case8_postponed",
                "title": "Call vendor and sign studio lease renewal",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_col3",
                "message_id": "m125",
                "acceptable_origin_ids": ["m125"],
                "due_date": "2026-11-05",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "remind"},
            },
            {
                "id": "rl_case9_cancelled",
                "title": "Change balcony light switch",
                "type": "followup",
                "owner": "p_vendor_elec",
                "chat_id": "c_dm_elec",
                "message_id": "m128",
                "acceptable_origin_ids": ["m128"],
                "due_date": "2026-10-27",
                "status": "dropped",
                "closure_message_id": "m129",
                "acceptable_closure_ids": ["m129"],
                "intervention": None,
            },
            {
                "id": "rl_case11_consent",
                "title": "Sign school science excursion consent form",
                "type": "deadline",
                "owner": "ME",
                "chat_id": "c_grp_class4b",
                "message_id": "m130",
                "acceptable_origin_ids": ["m130"],
                "due_date": "2026-11-02",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "remind"},
            },
            {
                "id": "rl_case15_elec",
                "title": "Pay October electricity bill via OTP",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_grp_home",
                "message_id": "m132",
                "acceptable_origin_ids": ["m132"],
                "due_date": "2026-11-03",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "remind"},
            },
            {
                "id": "rl_case15_broadband",
                "title": "Pay October broadband bill via OTP",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_grp_home",
                "message_id": "m133",
                "acceptable_origin_ids": ["m133"],
                "due_date": "2026-11-04",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "remind"},
            },
            {
                "id": "rl_case18_overdue",
                "title": "Submit society quarterly maintenance cheque",
                "type": "deadline",
                "owner": "ME",
                "chat_id": "c_grp_society",
                "message_id": "m135",
                "acceptable_origin_ids": ["m135"],
                "due_date": "2026-10-24",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "remind"},
            },
            {
                "id": "rl_case19_thirdparty",
                "title": "Rahul send staging server AWS logs to Sneha",
                "type": "followup",
                "owner": "p_colleague_1",
                "chat_id": "c_grp_zephyr_core",
                "message_id": "m137",
                "acceptable_origin_ids": ["m137"],
                "due_date": "2026-10-30",
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "leave", "leave_reason": "someone_elses"},
            },
            {
                "id": "rl_case20_private_prep",
                "title": "Shortlist two beachfront resort options in Coorg or Goa",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_spouse",
                "message_id": "m139",
                "acceptable_origin_ids": ["m139", "m140"],
                "due_date": None,
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "assist", "needs_approval": False},
            },
            {
                "id": "rl_personal_leave",
                "title": "Write personal anniversary note",
                "type": "commitment",
                "owner": "ME",
                "chat_id": "c_dm_spouse",
                "message_id": "m142",
                "acceptable_origin_ids": ["m142"],
                "due_date": None,
                "status": "open",
                "closure_message_id": None,
                "intervention": {"action": "leave", "leave_reason": "personal"},
            },
        ],
    }

    with (out / "chats.jsonl").open("w", encoding="utf-8") as fh:
        for c in chats:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    with (out / "messages.jsonl").open("w", encoding="utf-8") as fh:
        for m in messages:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    (out / "seed.json").write_text(
        json.dumps(seed_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "answers" / "reference.json").write_text(
        json.dumps(reference_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    p = build_synthetic_benchmark("tmp_synth_eval")
    print(f"Generated synthetic adversarial benchmark at {p}")
