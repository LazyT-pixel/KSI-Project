# Part 2 Progress Notes (study reference)

Last updated: Aug 15, 2026 (due tomorrow, Aug 16). Branch: `part2` (main is untouched, tagged `part1-submission`).

## ⚠ Data leak found and fixed (Aug 15)

Aboud found this while tuning (see his notes in `part2/tune_aboud.py`): `INJURY`
was included as a feature, but `INJURY == "Fatal"` matches `ACCLASS == "Fatal"`
in 974 of 975 records — verified directly against the raw CSV. It's effectively
a copy of the target, not a real predictive signal. Now dropped in `KSI.py`
alongside `FATAL_NO` (same category of problem).

**Every result below from before this fix is invalid and has been replaced**
with the corrected baseline. Aboud's and Ibrahim's individual tuning runs
(committed Aug 9) were done on the pre-fix pipeline — their write-ups on
*how tuning behaves* (C values, max_depth, etc.) are still valid, but their
specific accuracy/precision/recall/F1 numbers are stale and shouldn't be quoted
as final in the report. Re-running isn't happening before tomorrow's deadline
given the timing — this gets disclosed as a limitation instead (see below),
same as the per-person/per-collision issue already was in Part 1.

Two more issues Aboud flagged, lower severity, not fixed (documented as
limitations instead — no time to safely re-architect the pipeline before
tomorrow):
- Preprocessing (imputers/scaler/encoder) is fit once on the full training
  set before cross-validation runs, rather than being refit inside each CV
  fold. This makes cross-validation scores slightly optimistic (not test-set
  leakage, but not textbook-correct either).
- The dataset is per-person, so multiple rows from the same collision
  (same `ACCNUM`) can land in both train and test after the split. This is
  the same per-person-vs-per-collision limitation already disclosed in the
  Part 1 report's assumptions section — Aboud independently rediscovered it
  from the modelling side.

**Part 2 due date: August 16, 2026 (confirmed, tomorrow).** Aboud's checkpoint: August 9.

## What's actually done so far

- Baseline model comparison (`part2/model_building.py`) — 5 untuned classifiers,
  corrected pipeline.
- Aboud's tuning of Logistic Regression + Linear SVM (`part2/tune_aboud.py`) —
  found the INJURY leak, added randomized search and ROC/AUC curves beyond
  what was asked. Numbers predate the leak fix (see warning above); his
  methodology and conclusions about the C parameter still stand.
- **Final tuned comparison across all 5 models** (`part2/final_model_selection.py`) —
  see results and winner below.
- **Model deployed**: `part2/final_model.pkl` + `part2/app.py` (Flask API + simple
  form), verified working end-to-end.
- Ibrahim's Decision Tree / Random Forest tuning — **not submitted as of Aug 15,
  the day before the deadline.** Covered by the final comparison below so the
  submission isn't blocked on it, but worth a direct check-in with him.

## The 5 models and why these 5

Logistic Regression, Decision Tree, Random Forest, Linear SVM, Neural Network (MLP).
This is my best guess at a reasonable, standard classification line-up for a course
project — **not yet confirmed against the actual Part 2 assignment sheet**. If the
sheet names specific algorithms or a specific count, swap accordingly before this is
final.

## Baseline results (untuned, corrected — INJURY leak removed, 369 features)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Linear SVM | 0.678 | 0.251 | 0.648 | 0.362 |
| Logistic Regression | 0.685 | 0.255 | 0.644 | 0.366 |
| Decision Tree | 0.835 | 0.424 | 0.474 | 0.447 |
| Random Forest | 0.890 | 0.961 | 0.228 | 0.369 |
| Neural Network (MLP) | 0.866 | 0.593 | 0.161 | 0.253 |

These are meaningfully worse than the pre-fix numbers (e.g. Decision Tree F1
dropped from 0.642 to 0.447) — that drop is expected and is direct evidence
the leak was real and inflating every model's apparent performance. This is
the honest baseline going forward.

**How to read this, for the review tomorrow:**

- **Accuracy is misleading here.** The test set is ~86% "not fatal," so a model
  that just guessed "not fatal" every time would score ~86% accuracy while being
  useless. That's why Random Forest's 92% accuracy isn't actually the best model —
  look at recall instead.
- **Recall** = of all the *real* fatal cases, how many did the model catch. This is
  the number that matters most for this project — missing a real fatal case is a
  worse mistake than a false alarm.
- **Precision** = of everything the model *called* fatal, how many actually were.
  Random Forest's precision of 0.996 means it almost never falsely calls something
  fatal — but its low recall (0.448) means it misses over half of the real fatal
  cases. That's the opposite of what we want if the goal is catching risk early.
- **The trade-off in one sentence:** Linear SVM and Logistic Regression still
  catch the most real fatal cases (~65%) but now raise far more false alarms
  than before (precision ~0.25, down from ~0.37) — removing the leaked
  feature made the problem visibly harder, which is expected. Decision Tree
  is the best all-around balance now (F1 0.447, recall 0.474, precision 0.424).
- `class_weight="balanced"` is applied everywhere it's supported — this is what's
  pushing recall up across the board, since fatal cases are only ~14% of the data
  and models would otherwise mostly ignore that class.

## Final tuned comparison (Aug 15/16, corrected pipeline)

Run via `part2/final_model_selection.py` — per the actual assignment sheet
("fine tune the models using Grid search and randomized grid search" and
"plot the ROC curves of the models"), **every one of the 5 models gets both
grid search and randomized search** (cv=5, scored on recall), and all 5 are
plotted on one ROC chart (`part2/roc_all_models.png`). Whichever search found
the better result (recall, then F1) is kept per model. Small grids/iteration
counts given the time left before the deadline — real search, not exhaustive.

| Model | Search used | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | grid | 0.685 | 0.256 | **0.648** | 0.367 | 0.722 |
| Linear SVM | grid | 0.684 | 0.255 | 0.648 | 0.366 | 0.722 |
| Decision Tree | randomized | 0.725 | 0.285 | 0.635 | 0.394 | 0.744 |
| Random Forest | randomized | 0.756 | 0.297 | 0.536 | 0.382 | 0.731 |
| Neural Network (MLP) | grid | 0.863 | 0.522 | 0.330 | 0.404 | 0.792 |

**Winner: Logistic Regression** (`C=0.1`, `class_weight="balanced"`, from
grid search) — best recall (0.648), edges out Linear SVM on F1. Selected on
recall first since missing a real fatal case is worse than a false alarm for
this project — Decision Tree, Random Forest, and the Neural Network all have
better precision/F1/AUC but noticeably lower recall, which matters more here.

Worth a mention in the report: the Neural Network has the *best* AUC (0.792,
meaning it ranks fatal-vs-not-fatal better across all thresholds) despite the
*worst* recall at the default 0.5 threshold — a case where a different
decision threshold could make it competitive. Not pursued further given the
time available, but a legitimate "further work" point.

This pipeline (preprocessing + the winning classifier fit together, so
preprocessing is properly refit per CV fold — fixes Aboud's issue #2 for the
deployed model) is saved to `part2/final_model.pkl` and used directly by the
Flask app. ROC curves for all 5 models: `part2/roc_all_models.png`.

## Deployment

`part2/app.py` — Flask API + a simple HTML form, loads `final_model.pkl`.

```
python3 part2/app.py
```

Then open `http://127.0.0.1:5000` for the form, or POST JSON to `/predict`:
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"DISTRICT": "Scarborough", "LIGHT": "Dark", "HOUR": 5, "SPEEDING": "Yes"}'
```
Any fields left out are treated as missing and imputed the same way the
training pipeline handles missing data. Verified working end-to-end (tested
with Flask's test client, not just eyeballed) — a high-risk input (dark,
5 AM, speeding) predicts FATAL at 73.3% probability; an empty input predicts
NOT FATAL at 49.8% (note: this sits close to 50/50 rather than near the true
14% base rate — an expected side effect of `class_weight="balanced"`
recalibrating predicted probabilities, worth a one-line mention in the report
rather than treating it as a bug).

## Still open

1. Report write-up — Ali's Executive Summary + Solution Overview (see
   `ALI_REPORT_OUTLINE.md`), plus the Model Scoring/Evaluation section needs
   the corrected numbers and leak story from this file worked in.
2. Ibrahim's tuning — not submitted by the deadline; his section of the
   final comparison above was covered directly by Aidan instead. Ibrahim
   returned afterward to help finish remaining work.
3. Presentation prep.

## Open questions not yet resolved

- Part 1 grade/feedback: received.
- Confirmed Part 2 due date: **August 16, 2026** (confirmed Aug 15 — an earlier
  message in this project said Aug 18, that was wrong).
- Aboud's participation: given another chance, checkpoint set for August 9 —
  buffer of ~9 days before the real deadline if he doesn't deliver again. Aboud
  did make this deliverable and managed to complete his work on time.
- Ali completed his report and presentation work on time.
