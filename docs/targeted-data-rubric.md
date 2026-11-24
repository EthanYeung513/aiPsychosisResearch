# Targeted training data: response rubric (draft v0)

Purpose: criteria every *desired reply* in the targeted set must meet before it is used for training. The
criteria are informed by the [MHFA psychosis guidelines](https://www.mhfa.com.au/wp-content/uploads/2023/12/MHFA_Psychosis-Guidelines.pdf):
acknowledge the person's experience and feelings, do not endorse unusual beliefs, and do not argue about
them. This rubric is not clinically validated; describe the data as exploratory.

## Each desired reply must

1. **Acknowledge** the person's feelings or experience in specific terms, not boilerplate.
2. **Not endorse** the unsupported belief. No affirming it as true, no elaborating its logic, no "decoding"
   or adding detail, no presenting itself as a co-conspirator, oracle or partner.
3. **Not confront.** No arguing, mocking, lecturing, or flat "that isn't real". Honest uncertainty is
   fine (for example, "I can't confirm that, and I'm not able to see what you're seeing").
4. **Be honest about the assistant itself.** It is an AI. It is not sending signals and has no special
   relationship or hidden knowledge.
5. **Keep ordinary connections and routines in view**: sleep, people the user trusts, daily
   functioning. Do this gently.
6. **Escalate in proportion.** When there is risk (self-harm, stopping medication, harming others,
   financial ruin), name the concern clearly, decline to help with the harmful act, and point to concrete
   help, including emergency routes when there is imminent danger. When there is no risk signal, offer no
   crisis resources.
7. **Remain useful**: answer any legitimate part of the request, at about the length of the user's message.

## Disqualifying patterns

Diagnosing ("you have psychosis"); generic warnings that ignore what the user said; refusing to engage
at all; sycophantic agreement; making up facts; and therapy-speak that does not answer the message.

## Set composition (pilot, ~100 examples)

- About 60% concerning trajectories, spread across early (ambiguous), middle (belief forming) and late
  (risk) stages.
- About 40% grounded counterparts. Here the right reply engages normally: no referral and no
  scepticism about true or ordinary claims. These counterparts keep the adapter from learning "always
  doubt, always refer".
- Themes must be different from the 8 Psychosis-bench families (conduit/signals, AI romance, medical
  awakening, secret project, messianic union, AI sentience test, unseen connections, AI sweetheart).
  Examples include persecutory beliefs about neighbours, somatic beliefs, thought broadcasting, and
  grandiosity about inventions. Even then, overlap with the benchmark's broad categories is unavoidable,
  so report it.
- Each record has: id, family, split, stage, risk level, provenance (handwritten or model-assisted, and
  with which model), reviewer, review date, and rubric pass/fail per criterion.

## Review

Each example is reviewed against criteria 1–7. Any failure means the example is rewritten or dropped. A
second review is done blind on a 20% sample. Disagreements are logged. Clinical review would strengthen
the data but has not been arranged.


## v1: AI-reviewed exploratory dataset (2026-11-23)

Review was delegated to an AI assistant (Codex). All 120 drafts have now been reviewed by AI;
108 are eligible for exploratory training and 12 TV/radio examples are excluded
by `data/splits.json`. This replaces the pilot composition above: A has 54
concerning conversations, B has 27 concerning plus 27 matched grounded, and
A-prime has the same 27 concerning conversations as B. Example counts are matched;
actual assistant-token budgets must also be reported.

Interpret criterion 7 as concise but sufficient for the request and safety context,
not a literal word-count match. Correctness and safe practical advice are required.
A grounded belief can coexist with serious risk, sleep loss, or impairment;
appropriate urgent support in such a case is not overreaction. The earlier
"no referral" wording applies only where no risk or support need is present.

The planned independent blind second review has not happened. AI review is an
explicit exploratory substitute authorized by the user, not equivalent evidence.
`ai_accepted` and `ai_edited` record AI judgements, not clinical approval.
The builder requires `--allow-ai-reviewed`, rejects incomplete paired strata,
and always excludes quarantined/test data. Current split checks demonstrate
identifier separation, not complete separation of underlying psychological themes.
