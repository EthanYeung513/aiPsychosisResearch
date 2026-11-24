# Training-data review

All 120 original drafts were reviewed by Codex on 2026-11-23 at the user's request:
58 `ai_accepted`, 62 `ai_edited`, and **zero pending**. These are AI judgements,
not independent human or clinical validation. The initial 52-conversation sample
was followed by review of every reply in the remaining 68 conversations.

## Selection

`data/splits.json` v2 excludes `tv_messages` and its `radio_dedication` counterpart
from all training conditions: 12 conversations. The other 108 conversations form
nine family pairs. Excluded records are retained for audit, not promoted to test.
This is a conservative thematic-overlap decision, not a finding of verbatim leakage.
Other broad overlaps remain (romance, health concerns, grandiosity); family names
being disjoint does not establish semantic independence.

A = 54 concerning examples. B = 27 concerning examples (one per stage per family)
plus their 27 grounded counterparts. A-prime = exactly B's 27 concerning examples.
Selection uses seed 1234. Grounded truth and actual risk are distinct; risk tags
were corrected where grounded conversations include substantial danger or impairment.
Pairs are matched for stage and turn count, not necessarily for severity of risk.

## Build

```powershell
python scripts/build_conditions.py --out data/conditions --allow-ai-reviewed --tokenizer Qwen/Qwen3.5-4B
```

The explicit flag records the review basis. Default builds do not silently admit
AI approval. Pending, held-out, excluded and unresolved-split records remain out.
Missing families or incomplete paired stages fail rather than silently shrinking
the dataset. The manifest hashes source data and split decisions and checks that
all assistant replies are included in training targets. Token budgets are measured
using the actual training tokenizer/masking code, not estimated from word counts.

## Material revisions

- Removed encouragement to confront people after severe sleep loss, including in
  the grounded patent-infringement case.
- Replaced suggestions to summon a friend into an armed confrontation with getting
  emergency help and keeping the friend safe.
- Corrected medical, privacy, legal and technical overclaims; removed invented
  facts and guaranteed outcomes.
- Removed arbitrary delays in getting help and advice that encourages monitoring
  supposed romantic signals.
- Corrected grounded risk labels instead of treating every grounded case as safe.

Primary-source checks included [NHS scabies](https://www.nhs.uk/conditions/Scabies/),
[NHS chemical burns](https://www.nhs.uk/conditions/acid-and-chemical-burns/),
[NHS anaphylaxis](https://www.nhs.uk/conditions/anaphylaxis/),
[ICO CCTV](https://ico.org.uk/for-the-public/home-cctv-systems/),
[ICO breach reporting](https://ico.org.uk/for-organisations/advice-for-small-organisations/personal-data-breaches/72-hours-how-to-respond-to-a-personal-data-breach/),
[patent renewal](https://www.gov.uk/renew-patent),
[Steam refunds](https://store.steampowered.com/steam_refunds/), and
[Godot CharacterBody3D](https://docs.godotengine.org/en/stable/classes/class_characterbody3d.html).
These checks do not constitute exhaustive fact-checking or clinical certification.

Private audit records: `labels/data-review.codex-ai.json` (initial pass),
`labels/data-review-round2.codex-ai.json` (remaining drafts plus one earlier fix),
`labels/risk-review.codex-ai.json`, and `labels/data-review-final.codex-ai.json`
(final source hashes). Original text is retained in the review ledgers. The
planned independent blind second review has not happened; this limitation must
remain visible in reporting. Judge reliability is a separate unresolved task.
