# Official dev evaluation: 11 November 2026

The retained offline implementation scores **28.247735639442958/56**, an increase of
8.4483 points over the 19.7995 starting branch and 0.5977 above the supplied 27.65
benchmark. This is an iteratively tuned dev result, not held-out evidence or proof of
first place. Goals and intervention labels are absent; neither category was scored.
No submission, merge or new paid model call was made.

## Causes and measured corrections

| Cumulative correction | Official /56 | Predicted loops | Matched /74 |
|---|---:|---:|---:|
| Fresh starting branch | 19.7995 | 408 | 32 |
| External customer groups, family surname and business politeness | 20.9836 | 408 | 32 |
| Require grounding for unaddressed group requests | 21.5795 | 291 | 32 |
| Dominant household contact, workspace discovery and delivery-group routing | 23.0109 | 294 | 32 |
| Informal imperatives, proposals and promises | 24.2993 | 313 | 41 |
| Explicit employment/customer evidence and sibling specificity | 25.4700 | 313 | 41 |
| Reject sentence words as ME identity aliases | 25.8119 | 273 | 41 |
| Concrete needs and typed delivery evidence | 27.4074 | 278 | 45 |
| Route discovered flat groups; ground large-group promises | 27.6605 | 273 | 45 |
| Purpose-bearing document requests and explicit purchase authorisation | **28.2477** | **276** | **48** |

Each measured candidate passed the official validator/checker; aggregate scores use
unmodified `check.py`. A stricter alias experiment scored 25.8215 but broke two
identity regressions; it was rejected in favour of 25.8119 with those tests passing.
All retained changes pass the final 97-test suite, including official worked-example
and zero-seed example checks, both 74/74, and the shell packaging test.

The largest precision defect was assigning unaddressed operational group requests to
ME by default. A large operations group alone produced 90 baseline loops without
matching a labelled origin. This is a diagnostic, not a claim that all 90 are false:
group requests now require addressing, quotation, acceptance or a small household
context. Large-group promises require direct grounding in ME's conversation.

Identity extraction also admitted ordinary sentence words such as "you" and "also"
as ME aliases, which falsely satisfied the addressing guard. Customer delivery chats
containing the user's company brand were misread as internal coworker groups.
Generic "would love" language contributed spouse evidence and suppressed the
household-contact fallback. Explicit sibling groups now outrank weak family ambiguity;
company tokens are excluded from the added shared-surname prior.

Organisation discovery incorrectly required no known companies before creating an
office workspace. Staff can now support the workspace even when a company is seeded.
Flat/unit groups were detected during discovery but omitted from ownership routing;
the routing step now consumes that discovery evidence. Branded colon-format delivery
groups can route to the user's company while explicit client-company bridge rules
remain intact. This does not infer private company affiliations from sparse names.

Missed task origins were often punctuation-free imperatives, shorthand politeness,
concrete needs, proposals or requirements for an artifact. Extraction now recognizes
these speech acts and preserves the original source and speaker-based ownership.
Closure accepts an owner's prompt same-DM phone-number/map delivery for the matching
request, or a topic-linked explicit purchase confirmation. Ordinary links, unrelated
numbers, bare acknowledgements and negated purchases do not establish completion.
Expired due dates never establish completion. Duplicate reference source IDs do not
produce duplicate predictions.

## Public competitor mechanisms inspected

CodeInfinity1's [reviewed public revision](https://github.com/CodeInfinity1/cold-start/commit/b44849fde68f57b109a3bb3bbe2a2473ac1ade77)
reports 27.6 offline and 27.9 for a partial real-model run. Its rules use contact/company
tokens and acronyms, internal versus external groups, and surname context. These
mechanisms motivated inspecting our evidence conflicts; our fixes above are justified
by addressing and role semantics, with different synthetic identities in regressions.
Its write-up also notes reference completions without visible closing evidence;
we preserve evidence-required closure rather than exploiting elapsed dates.

The [reviewed aryanpal2006 revision](https://github.com/aryanpal2006/Xpert-Intern-task/commit/2c2262fa4325fb4010290e97ea2b28a36aab4697)
explicitly replaces its previous identity-specific and duplicated-loop implementation
with general model inference and structural fallback. Earlier inflated results are
not a credible comparable benchmark. No competitor answers, IDs or names were added
to inference, and no external implementation was copied.

## Final measurements and limits

| Metric | Starting branch | Retained version |
|---|---:|---:|
| Circles /12 | 6.4390 | 9.0732 |
| Organisations /8 | 2.3222 | 3.6958 |
| Chats /8 | 6.3415 | 6.3415 |
| Loops found /14 | 1.8589 | 3.8400 |
| Loops closed /14 | 2.8378 | 5.2973 |
| Label-based precision | 7.84% | 17.39% |
| Label-based recall | 43.24% | 64.86% |
| Label-based F1 | 13.28% | 27.43% |
| False DONE among matched loops | 0 | 0 |
| Incorrect owner among matched loops | 5/32 | 6/48 |
| Pipeline runtime (single local run) | 11.0947 s | 6.7569 s |

Runtime is observational, not a controlled latency benchmark. Final tokens: **0**;
model requests: **0**; API cost: **$0**. OpenRouter was deliberately not retried after
the earlier permanent-credit failure. Provider-backed quality is still unverified
for these changes. Private datasets, predictions, diagnostics and credentials remain
outside tracked files. The inference pipeline never reads reference answers.

Remaining weaknesses: 26 unmatched reference loops, 20 matched missed completions,
42 missed completions including undiscovered loops, six matched owner errors,
organisation member/ownership ambiguity and ten incorrect unseeded circles out of
41 labels. Some reference origins are fragments, media or duplicated sources, and
some completed labels lack visible closure. We have not treated those as rules to
memorize, nor treated unmatched predictions as automatically false. The dev score
alone cannot establish reliability on the hidden set or global competitor rank.
