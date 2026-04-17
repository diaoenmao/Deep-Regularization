# Venue Strategy (2026-03-26)

## Current Recommendation

Primary target: `TMLR`

Secondary target: `DMKD`

Current judgment:

- If the paper is submitted in its current state, `TMLR` is the better fit.
- If the paper is cleaned up substantially and rewritten with a stronger data-mining framing, `DMKD` becomes a serious alternative.
- `TKDE` is not the best next target for the current manuscript. The paper is not yet clean enough on protocol, traceability, and claim discipline to make that the first choice.

## Why `TMLR` Is the Better First Target

`TMLR` is currently the best match for the paper's development stage, not necessarily because the topic is uniquely TMLR-shaped, but because its review model is well aligned with the paper's main remaining work: tightening claims, fixing protocol mismatches, and improving traceability.

The official TMLR policies explicitly say that acceptance is based on whether the claims are supported by accurate, convincing, and clear evidence, and whether some part of the TMLR audience would care about the findings. They also state that papers should be accepted even when the contribution is modest, as long as those criteria are met. This is a good fit for a paper whose main risk is rigor rather than lack of topic relevance.

TMLR also uses an open-review, double-blind process with an iterative rebuttal and revision phase. That is useful here because the manuscript still has several fixable but material issues documented in the paper audit: overbroad real-world claims, paper/code mismatch in the main predictor architecture, and incomplete raw-result provenance.

TMLR is also operationally attractive:

- rolling submission,
- no page limit in the conference sense,
- explicit encouragement to submit code and data for reproducibility,
- possible Journal-to-Conference presentation path for selected papers.

## Why `DMKD` Remains a Strong Backup

`Data Mining and Knowledge Discovery` is probably the closest venue by topic. Its official aims and scope explicitly cover data mining methods, optimization, classification, data selection and reduction, evaluation, and knowledge discovery workflows. That matches this paper's actual content very well: feature selection, optimization, benchmark evaluation, and downstream validation.

If the paper is revised into a more data-mining-centered narrative, `DMKD` may become the more natural venue by scope. In practice, that would mean:

- reducing conference-style rhetoric,
- framing the contribution more as a feature-selection and empirical methodology paper,
- making the reproducibility and benchmark protocol section more explicit,
- softening any broad "state-of-the-art" style language.

The reason `DMKD` is still second rather than first is practical: the current manuscript still reads more like a cleaned-up conference paper than a fully settled journal article. `TMLR` tolerates that transition state better.

## Side-by-Side Comparison

| Venue | Best fit right now | Best fit after full cleanup | What the paper must do well | Main risk for this paper |
|---|---|---|---|---|
| `TMLR` | Strong | Strong | precise claims, evidence discipline, reproducibility, clear ML contribution | open review will expose any unresolved inconsistency immediately |
| `DMKD` | Moderate | Strong | benchmark rigor, data-mining framing, practical feature-selection contribution | paper may still read too much like an ML conference submission |
| `Machine Learning` | Moderate | Moderate to strong | clean methodological framing, solid empirical support, clear learning contribution | less directly aligned with the data-mining/feature-selection angle than `DMKD` |
| `TKDE` | Weak to moderate | Moderate | stronger engineering/data systems framing and very polished evidence chain | current manuscript maturity is likely below the bar for first-choice submission |

## Venue-Specific Read on This Manuscript

### `TMLR`

What works:

- the paper proposes a concrete learning/optimization method,
- the empirical story is broad enough for ML readers,
- the strongest revision need is evidence quality and claim calibration, which TMLR explicitly centers.

What must be fixed before submission:

1. Remove every sentence implying SADMM-FS matches or exceeds RF on all nine datasets.
2. Resolve the `2x32` vs `5x58` code/paper mismatch.
3. Produce a committed result manifest for every table and figure.
4. Clarify that architecture sweeps are internal ablations, not cross-method tuning comparisons.
5. Tighten the real-world protocol description around predefined splits vs seeded random splits.
6. Add explicit reproducibility pointers in the submission package.
7. Add the required first-page footnote if LLM tools were used in writing.

### `DMKD`

What works:

- the paper topic is highly aligned with the journal scope,
- the benchmark-and-feature-selection story is natural here,
- the downstream selection-quality framing fits the journal.

What would need to be rewritten beyond the TMLR fixes:

1. Reframe the introduction around data mining and knowledge discovery rather than conference-style ML novelty positioning.
2. Reduce emphasis on "first among neural methods" unless the comparison protocol is fully controlled and justified.
3. Strengthen the problem framing around feature selection under nonlinear interactions and high-decoy regimes.
4. Expand the benchmark protocol and dataset discussion so the paper reads as a durable empirical reference, not just a method paper.

## Recommended Decision

Use the following decision rule:

- If the goal is the fastest path to a defensible, internationally credible submission after a serious cleanup, choose `TMLR`.
- If the paper becomes substantially more stable, more traceable, and more explicitly data-mining-oriented during revision, keep `DMKD` as the fallback or second submission target.

Today, the paper should be prepared for `TMLR`.

## TMLR Revision Roadmap

### Phase 1: Eliminate hard factual and fairness problems

This phase is mandatory. The paper should not be submitted before these are fixed.

1. Rewrite the real-world headline claim.
   - Remove all variants of "matches or exceeds RF on all nine datasets."
   - Replace with narrower claims already supported by the table.

2. Resolve the architecture mismatch.
   - Either make the committed benchmark code use the stated fixed `2x32` predictor and rerun affected results.
   - Or revise the paper so the text accurately describes the architecture that produced the reported numbers.
   - The first option is better.

3. Build a result manifest.
   - Each table and figure should map to exact committed source files.
   - Missing SADMM-FS/STG raw JSONs for the real-world section need to be regenerated or restored.

4. Fix protocol wording.
   - Real-world splits must be described as predefined when available, seeded random otherwise.
   - Architecture sweeps must remain explicitly internal to SADMM-FS.

### Phase 2: Make the reproducibility story explicit

This is where the paper becomes "journal-ready" rather than just "conference paper plus appendix."

1. Add a short reproducibility paragraph to the experiments section.
   - state fixed seeds,
   - report which results are averaged over seeds,
   - point to code and raw artifacts.

2. Create a compact appendix or repository manifest documenting:
   - benchmark entry points,
   - dataset split handling,
   - result file naming conventions,
   - plotting dependencies.

3. Verify that every figure in the paper is regenerable from committed inputs.

### Phase 3: Recenter the paper around defensible claims

The paper currently has enough results to make a strong but narrower contribution. The framing should follow the strongest evidence rather than the most ambitious possible story.

Recommended claim hierarchy:

1. Primary claim:
   - `SADMM-FS` is a competitive neural feature-selection framework that is particularly strong on nonlinear interaction-heavy settings and high-decoy regimes.

2. Secondary claim:
   - The method's strength comes from the ADMM-based decoupling of gate optimization and sparsity enforcement, rather than from aggressive architecture tuning.

3. Controlled-ablation claim:
   - Predictor architecture matters for `SADMM-FS`, but architecture sweeps are an internal ablation, not benchmark-wide evidence.

Claims to avoid:

- universal superiority language,
- statements that imply all baselines were retuned equally when they were not,
- any wording that overstates the real-world table.

### Phase 4: Package the paper for TMLR specifically

1. Convert the manuscript to the TMLR template.
2. Keep the main body disciplined; long derivations can remain in the appendix.
3. Prepare anonymized supplementary material:
   - code or a code snapshot,
   - raw result files or compact reproducibility bundle,
   - plotting scripts if stable.
4. Add the required first-page footnote if LLM tools were used.
5. Prepare for OpenReview discussion:
   - have a concise "what changed" log ready,
   - be ready to respond with exact source-file references.

## Practical Next Step

The next highest-leverage move is not more writing. It is to close the paper/code mismatch and raw-result traceability gap first. Once those are fixed, the TMLR path becomes substantially cleaner.

## Sources

- TMLR editorial policies: https://www.jmlr.org/tmlr/editorial-policies.html
- TMLR author guide: https://www.jmlr.org/tmlr/author-guide.html
- TMLR FAQ: https://www.jmlr.org/tmlr/faq.html
- DMKD aims and scope: https://link.springer.com/journal/10618/aims-and-scope
- DMKD submission guidelines: https://link.springer.com/journal/10618/submission-guidelines
- Machine Learning aims and scope: https://link.springer.com/journal/10994/aims-and-scope
- TKDE call for papers / scope: https://www.computer.org/digital-library/journals/tk/cfp-ieee-transactions-on-knowledge-data-engineering
- Journal-to-Conference track: https://icml.cc/public/JournalToConference
