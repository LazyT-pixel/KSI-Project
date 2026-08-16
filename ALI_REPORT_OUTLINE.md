# Ali's Task — Report Writing (Part 2)

You're expanding two sections in `pipeline/report/Part1_Report.docx`: **Executive
Summary** and **Overview of Solution**. Both currently just have a placeholder
sentence saying "to be completed in Part 2" — you're replacing those with the
real thing.

## Executive Summary

Current placeholder text (for reference, don't just copy it — expand it):
> "This project builds a classifier that predicts whether a Toronto traffic
> collision will result in a fatality, using the Toronto Police KSI (Killed or
> Seriously Injured) dataset (2006-2023, ~19,000 records)."

This section is usually the shortest in the report (half a page or so) and is
often the only part some readers actually read closely — so it should stand on
its own. Cover, briefly:

1. What the project does (1-2 sentences — the placeholder above is a fine start)
2. What data it's built on (already have this — Toronto Police KSI dataset)
3. What was done technically, one sentence each: cleaned/prepared the data,
   compared 5 classification models, tuned the best ones, deployed the winner
   as an API
4. The headline result — once tuning is finished, this is the one number that
   matters most: the best model's recall on catching real fatal cases (not
   accuracy — see the note on that below)
5. One sentence on why this matters (identifying likely-fatal collision
   conditions could inform where traffic safety resources go)

**Wait to write point 4 until tuning results are in from Aboud/Ibrahim/me** —
everything else you can draft now.

## Overview of Solution

Current placeholder text:
> "Planned approach: clean and encode the KSI dataset, handle class imbalance,
> train and tune logistic regression, decision tree, SVM, random forest, and
> neural network classifiers, select the best performer, and deploy it behind
> a Flask API with a simple front end for inference."

This section explains *how* the project works, in more detail than the
Executive Summary, for a reader who wants the full picture without reading
the whole report. Suggested structure:

1. **Data preparation** — brief summary of the cleaning/encoding pipeline
   (already fully documented in the "Feature Selection" and "Data Modelling"
   sections later in the same report — you can summarize from those rather
   than re-deriving anything)
2. **Modelling approach** — the 5 models being compared (Logistic Regression,
   Decision Tree, Random Forest, Linear SVM, Neural Network), why 5 different
   types (different models catch different kinds of patterns, comparing them
   is more reliable than picking one blind), and that they're evaluated on
   recall (catching real fatal cases) rather than raw accuracy, because the
   fatal class is a small minority (~14%) of the data
3. **Model selection** — once tuning is done, name the winning model and
   why it was chosen
4. **Deployment** — the winning model gets saved (pickled) and served through
   a small Flask API, so it can take in details about a collision (location,
   time, road/weather conditions, etc.) and return a fatality-risk prediction

## Where to get supporting facts

- **Baseline model comparison table + explanation of accuracy vs. recall**:
  `PART2_PROGRESS.md` in the repo root — has the numbers and plain-language
  explanation of why recall matters more than accuracy here
- **Dataset details, feature choices, cleaning decisions**: already written up
  in the "Data Exploration and Findings" and "Feature Selection" sections of
  `Part1_Report.docx` itself
- **Final tuned results**: not available yet — Aboud and Ibrahim are working
  on this now (due Aug 9), I'm doing the Neural Network side

## Timeline

No rush on this specific file today — start drafting the parts that don't
depend on tuning results now if you want, but the numbers won't be final
until after Aug 9.
