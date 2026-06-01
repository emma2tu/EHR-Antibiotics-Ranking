The preprocessing pipeline is as follows: 
1. AMR_extract.py: 
- Get the antibiotic susceptible cohort from the MIMIC-IV ED dataset
- Filters microbiology cultures to Staph 
- Outputs `antibiotic_labels.csv` with 11 antibiotic columns (1 = susceptible, 0 = resistant).
2. Feature Engineering.py: 
- Converts raw MIMIC-IV-ED tables into textual format 
- Produces arrival (demographics + transport), vitals (all triage fields, "was not recorded" if missing), medrecon, pyxis, and ICD codes text fields. 
- Outputs `text_repr.json` keyed by stay_id.
3. data.py: 
- Merges text_repr.json with edstays.csv (to get hadm_id) and antibiotic_labels.csv. 
- Fills missing text fields with default sentences. 
- Outputs final `antibiotics.csv` for modeling.
