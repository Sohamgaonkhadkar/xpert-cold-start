# Ten-minute correction walkthrough

1. Two minutes: explain the immutable evidence store, contact dossiers, ARR closure
   trails and asymmetric intervention decisions. State measured dev limitations.
2. Two minutes: run `python3 -m scripts.correction_demo`. It uses only the public
   example, forces offline mode and asserts relationship/ownership/completion changes
   while unrelated people, goals and loops stay stable.
3. Three minutes: build a structured correction using exact IDs:

```bash
python3 -m src.corrections --file private-reports/corrections.json --person-circle P07 client
bash run.sh example private-reports/corrected.json --corrections private-reports/corrections.json
```

Read the public example output to choose an actual ID before using this template.
For a task, use its originating `source_msg`; internal `loop_...` IDs may change if
discovery changes. Add `--loop-owner SOURCE PERSON` or
`--loop-status SOURCE done --closure-msg EVIDENCE`. Closing evidence must be a real
later input message. A user correction asserts its relevance; the program validates
identity and time, but cannot prove the user's semantic claim.

4. Three minutes: show the correction regression tests and the isolated downstream
   intervention recalculation. Explain that corrections do not teach a global rule,
   rerun model inference or change unrelated records. Ambiguous person names or task
   title substrings fail clearly; use exact IDs to resolve them.

The demonstration intentionally changes public predictions for illustration. Never
present that corrected example as an improvement measured against hidden labels.
