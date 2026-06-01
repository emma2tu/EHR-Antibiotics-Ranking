import pandas as pd
import json
import os

ed_path = './mimic-iv-ed/2.2/'
hosp_path = './mimic-iv-ed/2.2/'

# ── load core tables ──────────────────────────────────────────────────────────
edstays = pd.read_csv(os.path.join(ed_path, 'edstays.csv'))
triage  = pd.read_csv(os.path.join(ed_path, 'triage.csv'))
medrecon = pd.read_csv(os.path.join(ed_path, 'medrecon.csv'))
pyxis   = pd.read_csv(os.path.join(ed_path, 'pyxis.csv'))
diagnosis = pd.read_csv(os.path.join(ed_path, 'diagnosis.csv'))
patients = pd.read_csv(os.path.join(hosp_path, 'patients.csv'))
admissions = pd.read_csv(os.path.join(hosp_path, 'admissions.csv'))

# ── merge demographics into edstays ──────────────────────────────────────────
edstays['intime'] = pd.to_datetime(edstays['intime'])
edstays = edstays.merge(patients[['subject_id', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='left')
edstays['age'] = (edstays['intime'].dt.year - edstays['anchor_year'] + edstays['anchor_age']).round().astype('Int64')
edstays = edstays.merge(
    admissions[['hadm_id', 'marital_status', 'insurance', 'language']].drop_duplicates('hadm_id'),
    on='hadm_id', how='left'
)

# ── arrival text ──────────────────────────────────────────────────────────────
# data.py parses subject_id as: str.split().str[1].replace(",","")
# so format must be "Patient {subject_id}, ..."
def make_arrival(row):
    age = f"{row['age']} year old " if pd.notna(row.get('age')) else ""
    race = row['race'].lower() if pd.notna(row.get('race')) else "unknown race"
    gender = {'M': 'male', 'F': 'female'}.get(str(row.get('gender', '')).upper(), 'unknown gender')
    transport = row['arrival_transport'].lower() if pd.notna(row.get('arrival_transport')) else None
    transport_str = f"via {transport} " if transport else ""
    marital = row['marital_status'].lower() if pd.notna(row.get('marital_status')) else 'unknown'
    insurance = row['insurance'].lower() if pd.notna(row.get('insurance')) else 'unknown'
    language = row['language'].lower() if pd.notna(row.get('language')) else 'unknown'
    return (
        f"Patient {row['subject_id']}, a {age}{race} {gender}, "
        f"arrived {transport_str}at {row['intime']}. "
        f"The patient's marital status is {marital}. "
        f"The patient's insurance is {insurance}. "
        f"The patient's language is {language}."
    )

edstays['arrival_text'] = edstays.apply(make_arrival, axis=1)

# ── vitals text (from triage) ─────────────────────────────────────────────────
def make_vitals(row):
    def fmt(val, name, unit=""):
        return f"{name} was {val}{unit}" if pd.notna(val) else f"{name} was not recorded"

    parts = [
        fmt(row.get('temperature'),  'temperature',              ' F'),
        fmt(row.get('heartrate'),    'pulse',                   ' bpm'),
        fmt(row.get('resprate'),     'respirations',            ' breaths/min'),
        fmt(row.get('o2sat'),        'o2 saturation',           '%'),
        fmt(row.get('sbp'),          'systolic blood pressure', ' mmHg'),
        fmt(row.get('dbp'),          'diastolic blood pressure',' mmHg'),
        fmt(row.get('pain'),         'pain score',              ''),
        fmt(row.get('acuity'),       'acuity level',            ''),
    ]
    if pd.notna(row.get('chiefcomplaint')):
        parts.append(f"chief complaint: {row['chiefcomplaint']}")

    return "At triage: " + ", ".join(parts) + "."

triage['vitals_text'] = triage.apply(make_vitals, axis=1)
vitals_map = triage.set_index('stay_id')['vitals_text']

# ── medrecon text ─────────────────────────────────────────────────────────────
def agg_medrecon(group):
    meds = group['name'].dropna().unique().tolist()
    if not meds:
        return None
    return "Medication reconciliation: " + "; ".join(meds) + "."

medrecon_map = medrecon.groupby('stay_id').apply(
    agg_medrecon,
    include_groups=False
)

# ── pyxis text ────────────────────────────────────────────────────────────────
def agg_pyxis(group):
    meds = group['name'].dropna().unique().tolist()
    if not meds:
        return None
    return "Pyxis medications dispensed: " + "; ".join(meds) + "."

pyxis_map = pyxis.groupby('stay_id').apply(
    agg_pyxis,
    include_groups=False
)

# ── ICD codes text ────────────────────────────────────────────────────────────
def agg_codes(group):
    codes = group['icd_code'].dropna().unique().tolist()
    if not codes:
        return None
    return "Diagnostic codes: " + ", ".join(codes) + "."

codes_map = diagnosis.groupby('stay_id').apply(
    agg_codes,
    include_groups=False
)

# ── assemble JSON keyed by stay_id ────────────────────────────────────────────
records = {}

for _, row in edstays.iterrows():
    sid = int(row['stay_id'])

    records[sid] = {
        'arrival': row['arrival_text'],
        'vitals': vitals_map.get(sid, None),
        'medrecon': medrecon_map.get(sid, None),
        'pyxis': pyxis_map.get(sid, None),
        'codes': codes_map.get(sid, None),

        # kept as None so data.py drop calls don't error
        'admission': None,
        'discharge': None,
        'eddischarge': None,
        'eddischarge_category': None,
    }

output_path = './text_repr.json'

with open(output_path, 'w') as f:
    json.dump(records, f)

print(f"Saved {len(records)} stays to {output_path}")