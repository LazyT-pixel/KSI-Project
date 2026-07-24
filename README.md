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
│   ├── 01_exploration_group1_location.ipynb          # Aboud — stub, pending
│   ├── 02_exploration_group2_people_vehicles.ipynb    # Ibrahim — stub, pending
│   ├── 03_exploration_group3_flags.ipynb              # Aidan — done, real findings
│   └── 04_data_modelling.ipynb                        # Aidan — Group 4, mirrors KSI.py with commentary
└── pipeline/
    ├── requirements.txt
    └── report/
        └── Part1_Report.docx     # Part 1 submission report
```

## Status (as of tonight)

- [x] Dataset downloaded, verified against glossary (52 columns match)
- [x] Target column (`ACCLASS`) definition and edge cases identified
- [x] Modelling pipeline built (`KSI.py`) — cleaning, column transformers,
      train/test split, imbalance-handling notes
- [x] Group 3 exploration (Aidan) — done, real stats + chart in
      `notebooks/03_exploration_group3_flags.ipynb`
- [x] Part 1 report drafted (`pipeline/report/Part1_Report.docx`) — includes
      Group 3 findings in full; Group 1/2 sections marked as placeholders
- [ ] Group 1 exploration (Aboud) — stub notebook ready at
      `notebooks/01_exploration_group1_location.ipynb`, needs real analysis
- [ ] Group 2 exploration (Ibrahim) — stub notebook ready at
      `notebooks/02_exploration_group2_people_vehicles.ipynb`, needs real analysis
- [ ] Once Group 1/2 findings are in: update the placeholder sections in
      `Part1_Report.docx` and re-check `KSI.py`'s column lists for anything
      that should be dropped (high missing %, unusable cardinality)
- [ ] Fill in Group #, Section #, and full team names on the report title page

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