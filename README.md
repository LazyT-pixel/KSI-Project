# COMP 247 Group Project — KSI Fatal Collision Predictor

Predicts whether a Toronto traffic collision will result in a fatality, using the Toronto
Police "Killed or Seriously Injured" (KSI) dataset. Built for COMP 247 — data
exploration, data modelling, predictive modelling, and deployment as a Flask API.

## Team & Roles (Part 1 — Week 10)

| Person   | Group | Focus |
|----------|-------|-------|
| Ali | —     | Initial dataset review, provided KSI glossary/metadata to the group (already complete) |
| Aboud    | 1     | Location & conditions columns |
| Ibrahim  | 2     | People & vehicles involved columns |
| Aidan       | 3 + 4 | Yes/No collision-type flags; data modelling pipeline |

## Dataset

- Source: [Toronto Police Public Safety Data Portal — Killed or Seriously Injured (KSI)](https://data.tps.ca/) — search "Killed and Seriously Injured", View Dataset → Download → CSV.
- Coverage: 2006–2023, 18,957 records (one row per person involved in a collision, not one row per collision).
- Full field glossary: see `docs/KSI_Glossary.pdf`.

### Target column
`ACCLASS` — classification of the record. Real values found in the data:
- `Fatal`
- `Non-Fatal Injury`
- `Property Damage Only` (stored truncated as `Property Damage O`)
- blank (dropped — no usable label)

**Assumption:** binary target = 1 if `Fatal`, else 0 (Non-Fatal Injury and Property
Damage Only both count as "not fatal"). Worth re-confirming as a group before Part 2.

### Column groups (52 total columns)

- **Group 1 — Location & conditions:** STREET1, STREET2, ROAD_CLASS, DISTRICT, LATITUDE, LONGITUDE, ACCLOC, TRAFFCTL, VISIBILITY, LIGHT, RDSFCOND, DIVISION, NEIGHBOURHOOD_158, HOOD_158, DATE, TIME
- **Group 2 — People & vehicles involved:** INVTYPE, INVAGE, INJURY, FATAL_NO, INITDIR, VEHTYPE, MANOEUVER, DRIVACT, DRIVCOND, PEDTYPE, PEDACT, PEDCOND, CYCLISTYPE, CYCACT, CYCCOND
- **Group 3 — 13 Yes/No collision flags:** PEDESTRIAN, CYCLIST, AUTOMOBILE, MOTORCYCLE, TRUCK, TRSN_CITY_VEH, EMERG_VEH, PASSENGER, SPEEDING, AG_DRIV, REDLIGHT, ALCOHOL, DISABILITY
- **Dropped (IDs/redundant):** INDEX, ACCNUM, OBJECTID, OFFSET, HOOD_140, NEIGHBOURHOOD_140 (old duplicate of _158), x, y (projected duplicate of LATITUDE/LONGITUDE)

## Repo structure

This is the actual current layout (differs slightly from earlier drafts — noted where relevant):

```
├── README.md
├── KSI.py                        # modelling pipeline script (cleaning/encoding/train-test/imbalance)
├── data/
│   └── TOTAL_KSI_4115794401574937330.csv   # raw dataset
├── docs/
│   └── KSI_Glossary.pdf          # field definitions from Toronto Police
├── notebooks/
│   ├── 01_exploration_group1_location.ipynb          # originally Aboud's — completed by Aidan, no submission received
│   ├── 02_exploration_group2_people_vehicles.ipynb    # Ibrahim — submitted, real analysis
│   ├── 03_exploration_group3_flags.ipynb              # Aidan — done, real findings
│   └── 04_data_modelling.ipynb                        # Aidan — Group 4, mirrors KSI.py with commentary
└── pipeline/
    ├── requirements.txt
    └── report/
        └── Part1_Report.docx     # Part 1 submission report
```

## Status — FINAL for Part 1 submission

- [x] Dataset downloaded, verified against glossary (52 columns match)
- [x] Target column (`ACCLASS`) definition and edge cases identified
- [x] Modelling pipeline built and verified running end-to-end (`KSI.py`) —
      cleaning, leakage/high-cardinality/structural-missingness columns
      dropped, HOUR feature engineered, encoding, 80/20 stratified split
      (15,164 train / 3,792 test rows, 373 features after encoding)
- [x] Group 3 exploration (Aidan) — done, real stats + chart in
      `notebooks/03_exploration_group3_flags.ipynb`
- [x] Group 1 exploration — originally assigned to Aboud; completed by Aidan
      after no submission was received by the internal deadline. Missing
      data/cardinality check, fatal-rate breakdowns, hour-of-day + district
      charts, in `notebooks/01_exploration_group1_location.ipynb`
- [x] Group 2 exploration (Ibrahim) — submitted just before the deadline.
      Full column-by-column analysis (data type, missing %, unique values,
      distribution, crosstab vs. ACCLASS) for every Group 2 column, in
      `notebooks/02_exploration_group2_people_vehicles.ipynb`. Verified it
      runs cleanly end-to-end; findings cross-validate with the rest of the
      report (e.g. pedestrian 18.2% vs. driver 12.7% fatal rate, matching
      the Group 2 numbers already in `Part1_Report.docx`)
- [x] Part 1 report finalized (`pipeline/report/Part1_Report.docx`) — all
      four sections complete, feature selection and data modelling sections
      reflect the final column decisions below
- [x] Per-person vs. per-collision decision: discussed, not implemented for
      Part 1 given the time available — documented as a Part 2 candidate
      improvement in the report's assumptions section instead of left open
- [x] Column decisions finalized in `KSI.py`: dropped `FATAL_NO` (data
      leakage — only populated when that person died), dropped
      `STREET1`/`STREET2` (too high-cardinality), dropped `DATE`/`TIME` in
      favor of an engineered `HOUR` feature, dropped the pedestrian/cyclist-
      only columns (structurally missing, redundant with Group 3's flags)

**Note on attribution:** Group 1's analysis was completed by Aidan after
Aboud did not submit work by the agreed internal deadline (see communication
timeline — Sunday planning session, Thursday soft deadline, Friday/Saturday
follow-ups). Group 2's analysis was submitted by Ibrahim directly, close to
the final deadline. This is disclosed here and in the report itself for
transparency ahead of peer evaluation.

## Setup

```bash
pip install -r pipeline/requirements.txt
```

## Notes / assumptions to double check as a group

- This is a per-person dataset, not per-collision — the same `ACCNUM` repeats
  for every person in a crash. Be careful not to accidentally treat rows as
  independent collision events when doing collision-level stats.
- Location fields are deliberately offset to the nearest intersection for
  privacy — not exact addresses.
- Dataset says 2006–2020 in the assignment doc but the portal now serves
  2006–2023 (updated since the assignment was written).