"""
Extract bleeding prediction cohort from MIMIC-IV CSV files
This script works directly with CSV files - no database needed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set your MIMIC-IV data path
MIMIC_PATH = os.path.expanduser('~/Downloads/MIMIC_IV')

print("="*60)
print("MIMIC-IV Bleeding Cohort Extraction")
print("="*60)
print(f"Data path: {MIMIC_PATH}\n")

# ============================================================================
# STEP 1: Load Core Tables
# ============================================================================

print("Step 1: Loading core tables...")

print("  Loading ICU stays...")
icustays = pd.read_csv(
    f'{MIMIC_PATH}/icu/icustays.csv',
    parse_dates=['intime', 'outtime']
)
print(f"    Total ICU stays: {len(icustays):,}")

print("  Loading patients...")
patients = pd.read_csv(f'{MIMIC_PATH}/hosp/patients.csv')
print(f"    Total patients: {len(patients):,}")

print("  Loading admissions...")
admissions = pd.read_csv(
    f'{MIMIC_PATH}/hosp/admissions.csv',
    parse_dates=['admittime', 'dischtime']
)
print(f"    Total admissions: {len(admissions):,}")

print("  Loading prescriptions...")
prescriptions = pd.read_csv(
    f'{MIMIC_PATH}/hosp/prescriptions.csv',
    parse_dates=['starttime', 'stoptime']
)
print(f"    Total prescriptions: {len(prescriptions):,}")

# ============================================================================
# STEP 2: Identify Anticoagulated Patients
# ============================================================================

print("\nStep 2: Identifying anticoagulated patients...")

# Anticoagulation medication keywords
anticoag_keywords = [
    'warfarin', 'heparin', 'enoxaparin', 'dalteparin',
    'dabigatran', 'rivaroxaban', 'apixaban', 'edoxaban',
    'aspirin', 'clopidogrel', 'ticagrelor', 'prasugrel'
]

# Filter prescriptions for anticoagulation
anticoag_rx = prescriptions[
    prescriptions['drug'].str.lower().str.contains(
        '|'.join(anticoag_keywords),
        na=False,
        case=False
    )
].copy()

print(f"  Anticoagulation prescriptions: {len(anticoag_rx):,}")

# Get unique hospital admissions with anticoagulation
anticoag_hadm_ids = anticoag_rx['hadm_id'].unique()
print(f"  Hospital admissions with anticoagulation: {len(anticoag_hadm_ids):,}")

# ============================================================================
# STEP 3: Build Cohort
# ============================================================================

print("\nStep 3: Building cohort...")

# Filter ICU stays to those with anticoagulation
cohort = icustays[icustays['hadm_id'].isin(anticoag_hadm_ids)].copy()

# Merge with patients
cohort = cohort.merge(patients, on='subject_id', how='left')

# Merge with admissions
cohort = cohort.merge(
    admissions[['hadm_id', 'race', 'admission_type']], 
    on='hadm_id', 
    how='left'
)

# Calculate age
cohort['age'] = cohort.apply(
    lambda x: (x['intime'].year - x['anchor_year']) + x['anchor_age'],
    axis=1
)

# Calculate length of stay in hours
cohort['los_hours'] = (cohort['outtime'] - cohort['intime']).dt.total_seconds() / 3600

# Get first anticoagulation time for each hospital admission, then map to stays
first_anticoag = anticoag_rx.groupby('hadm_id')['starttime'].min().reset_index()
first_anticoag.columns = ['hadm_id', 'anticoag_start']
cohort = cohort.merge(first_anticoag, on='hadm_id', how='left')

# Apply inclusion criteria
print(f"  Before filtering: {len(cohort):,} stays")

cohort = cohort[
    (cohort['age'] >= 18) &  # Adults only
    (cohort['los_hours'] >= 24)  # Minimum 24 hour stay
]

print(f"  After filtering: {len(cohort):,} stays")
print(f"  Unique patients: {cohort['subject_id'].nunique():,}")

# Save cohort
os.makedirs('data', exist_ok=True)
cohort.to_csv('data/cohort.csv', index=False)
print("  Saved to: data/cohort.csv")

# ============================================================================
# STEP 4: Identify Bleeding Events
# ============================================================================

print("\nStep 4: Identifying bleeding events...")

print("  Loading transfusion data...")
inputevents = pd.read_csv(
    f'{MIMIC_PATH}/icu/inputevents.csv',
    parse_dates=['starttime', 'endtime']
)

# PRBC transfusion itemids
prbc_itemids = [225168, 220970, 227070]

transfusions = inputevents[
    (inputevents['itemid'].isin(prbc_itemids)) &
    (inputevents['amount'] >= 1)
][['subject_id', 'hadm_id', 'stay_id', 'starttime', 'amount']].copy()

transfusions = transfusions.rename(columns={'starttime': 'transfusion_time'})

print(f"  Transfusion events: {len(transfusions):,}")

print("  Loading ICD diagnosis codes...")
diagnoses_icd = pd.read_csv(f'{MIMIC_PATH}/hosp/diagnoses_icd.csv')

# Bleeding ICD-10 codes
bleeding_icd_patterns = [
    'K92.0', 'K92.1', 'K92.2',  # GI bleeding
    'I60', 'I61', 'I62',  # Intracranial
    'R31', 'N02',  # Hematuria
    'R04',  # Respiratory bleeding
    'D62', 'R58'  # Other bleeding
]

# Filter for bleeding diagnoses
bleeding_diagnoses = diagnoses_icd[
    diagnoses_icd['icd_code'].str.startswith(tuple(bleeding_icd_patterns), na=False)
][['hadm_id', 'icd_code']].copy()

print(f"  Bleeding diagnoses: {len(bleeding_diagnoses):,}")

# Identify bleeding cases: transfusion AND bleeding ICD code
transfusion_stays = set(transfusions['stay_id'].unique())
bleeding_hadm_ids = set(bleeding_diagnoses['hadm_id'].unique())

cohort['has_transfusion'] = cohort['stay_id'].isin(transfusion_stays)
cohort['has_bleeding_icd'] = cohort['hadm_id'].isin(bleeding_hadm_ids)
cohort['bleeding'] = (
    cohort['has_transfusion'] & cohort['has_bleeding_icd']
).astype(int)

# Add bleeding time (first transfusion)
first_transfusion = transfusions.groupby('stay_id')['transfusion_time'].min()
cohort['bleeding_time'] = cohort['stay_id'].map(first_transfusion)

# Ensure bleeding occurs after anticoagulation start
invalid_bleeding = (
    (cohort['bleeding'] == 1) & 
    (cohort['bleeding_time'] < cohort['anticoag_start'])
)
cohort.loc[invalid_bleeding, 'bleeding'] = 0
cohort.loc[invalid_bleeding, 'bleeding_time'] = None

bleeding_count = cohort['bleeding'].sum()
bleeding_rate = bleeding_count / len(cohort)

print(f"\n  Bleeding cases: {bleeding_count}")
print(f"  Non-bleeding: {len(cohort) - bleeding_count}")
print(f"  Bleeding rate: {bleeding_rate:.1%}")

# Save updated cohort
cohort.to_csv('data/cohort.csv', index=False)
transfusions.to_csv('data/transfusions.csv', index=False)
bleeding_diagnoses.to_csv('data/bleeding_diagnoses.csv', index=False)

print("\n  Saved:")
print("    - data/cohort.csv")
print("    - data/transfusions.csv")
print("    - data/bleeding_diagnoses.csv")

# ============================================================================
# STEP 5: Extract Features (Sample)
# ============================================================================

print("\nStep 5: Extracting features (sampling for speed)...")

# For initial testing, sample a subset of patients
sample_size = min(1000, len(cohort))
cohort_sample = cohort.sample(n=sample_size, random_state=42)
sample_stay_ids = cohort_sample['stay_id'].tolist()

print(f"  Sampling {sample_size} stays for feature extraction")

print("  Loading vital signs (chartevents)...")
print("    Note: This file is LARGE (~25 GB). Loading will take several minutes...")
print("    We'll process in chunks to save memory...")

# Vital sign itemids
vital_itemids = {
    220045: 'heart_rate',
    220050: 'sbp',
    220051: 'dbp', 
    220052: 'map',
    220210: 'resp_rate',
    223761: 'temperature',
    220277: 'spo2',
    220739: 'gcs'
}

# Read chartevents in chunks and filter
vitals_list = []
chunk_size = 1_000_000

for chunk in pd.read_csv(
    f'{MIMIC_PATH}/icu/chartevents.csv',
    chunksize=chunk_size,
    parse_dates=['charttime'],
    usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum']
):
    # Filter for our cohort and vital signs
    chunk_filtered = chunk[
        (chunk['stay_id'].isin(sample_stay_ids)) &
        (chunk['itemid'].isin(vital_itemids.keys())) &
        (chunk['valuenum'].notna())
    ].copy()
    
    if len(chunk_filtered) > 0:
        chunk_filtered['feature_name'] = chunk_filtered['itemid'].map(vital_itemids)
        vitals_list.append(chunk_filtered)
    
    print(f"    Processed chunk... Found {len(chunk_filtered)} measurements")

vitals = pd.concat(vitals_list, ignore_index=True) if vitals_list else pd.DataFrame()

print(f"  Total vital measurements: {len(vitals):,}")
vitals.to_csv('data/vitals_sample.csv', index=False)
print("  Saved to: data/vitals_sample.csv")

print("\n  Loading lab values (labevents)...")
print("    Note: This file is also LARGE (~17 GB). Processing in chunks...")

# Lab itemids
lab_itemids = {
    51222: 'hemoglobin',
    51221: 'hematocrit',
    51265: 'platelet',
    51301: 'wbc',
    51237: 'inr',
    51275: 'aptt',
    50912: 'creatinine',
    51006: 'bun',
    50983: 'sodium',
    50971: 'potassium',
    50931: 'glucose'
}

sample_hadm_ids = cohort_sample['hadm_id'].tolist()

labs_list = []
for chunk in pd.read_csv(
    f'{MIMIC_PATH}/hosp/labevents.csv',
    chunksize=chunk_size,
    parse_dates=['charttime'],
    usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum']
):
    chunk_filtered = chunk[
        (chunk['hadm_id'].isin(sample_hadm_ids)) &
        (chunk['itemid'].isin(lab_itemids.keys())) &
        (chunk['valuenum'].notna())
    ].copy()
    
    if len(chunk_filtered) > 0:
        chunk_filtered['feature_name'] = chunk_filtered['itemid'].map(lab_itemids)
        labs_list.append(chunk_filtered)
    
    print(f"    Processed chunk... Found {len(chunk_filtered)} measurements")

labs = pd.concat(labs_list, ignore_index=True) if labs_list else pd.DataFrame()

# Add stay_id to labs
labs = labs.merge(cohort[['hadm_id', 'stay_id']], on='hadm_id', how='left')

print(f"  Total lab measurements: {len(labs):,}")
labs.to_csv('data/labs_sample.csv', index=False)
print("  Saved to: data/labs_sample.csv")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("EXTRACTION COMPLETE!")
print("="*60)

print("\nSummary:")
print(f"  Total cohort size: {len(cohort):,} ICU stays")
print(f"  Bleeding events: {bleeding_count} ({bleeding_rate:.1%})")
print(f"  Sample size for features: {sample_size}")
print(f"  Vital measurements: {len(vitals):,}")
print(f"  Lab measurements: {len(labs):,}")

print("\nFiles created in data/ directory:")
print("  - cohort.csv")
print("  - transfusions.csv")
print("  - bleeding_diagnoses.csv")
print("  - vitals_sample.csv")
print("  - labs_sample.csv")

print("\nNext steps:")
print("  1. Run: python create_sequences.py")
print("  2. This will create temporal sequences for model training")

print("\n" + "="*60)
