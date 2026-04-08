"""
Extract hierarchical ICD-10-CM features for the GRU bleeding prediction model.

Uses PRIOR admissions only to prevent data leakage — discharge diagnoses
from the current admission may contain codes assigned after the bleeding event.

Run AFTER extract_bleeding_cohort_anticoagulant.py (needs data/cohort.csv).

Outputs:
    data/icd_features.csv       — binary feature matrix (stay_id × icd_* columns)
    data/icd_vocabulary.json    — ordered list of selected hierarchical codes
"""

import pandas as pd
import numpy as np
import json
import os

# ---- Configuration ----
MIMIC_PATH = '/Users/nicholasperrotta/Downloads/MIMIC_IV'
MIN_FREQUENCY = 0.01   # Code must appear in >= 1% of cohort stays
MAX_FEATURES = 200     # Cap on number of ICD feature columns

# ICD-10-CM hierarchical level definitions (character-length → semantic name)
# Format: [Alpha][Numeric][AlphaNumeric] . [Etiology][AnatomicSite][Severity][Extension]
ICD10_LEVELS = {
    1: 'chapter',         # 1st char  — alpha chapter letter (A-T, V-Z)
    3: 'category',        # chars 1-3 — 3-character category (e.g. S32)
    4: 'etiology',        # char  4   — etiology detail       (e.g. S320)
    5: 'anatomic_site',   # char  5   — anatomic site         (e.g. S3201)
    6: 'severity',        # char  6   — severity              (e.g. S32010)
    7: 'extension',       # char  7   — extension character   (e.g. S32010A)
}


def normalize_icd10_code(code: str) -> str:
    """Strip dots and whitespace, uppercase an ICD-10 code."""
    return str(code).strip().replace('.', '').upper()


def hierarchical_propagate(code: str) -> set:
    """
    Expand an ICD-10-CM code into all hierarchical ancestor levels.

    ICD-10-CM structure (3-7 characters, dot removed):
        Pos 1       : Chapter letter        (Alpha, except U)
        Pos 1-3     : Category              (e.g. S32)
        Pos 1-4     : + Etiology            (e.g. S320)
        Pos 1-5     : + Anatomic site       (e.g. S3201)
        Pos 1-6     : + Severity            (e.g. S32010)
        Pos 1-7     : + Extension character (e.g. S32010A)

    Example:  S32.010A → S32010A
              propagated set = {S, S32, S320, S3201, S32010, S32010A}
    """
    code = normalize_icd10_code(code)
    levels = set()
    for length in ICD10_LEVELS:
        if len(code) >= length:
            levels.add(code[:length])
    return levels


# ============================================================================
print("=" * 60)
print("Hierarchical ICD-10 Feature Extraction")
print("=" * 60)

# ---- Step 1: Load cohort ----
print("\nStep 1: Loading cohort...")
cohort = pd.read_csv('data/cohort.csv')
print(f"  Cohort size: {len(cohort)} ICU stays")
print(f"  Unique patients: {cohort['subject_id'].nunique()}")

cohort_stay_ids = set(cohort['stay_id'].unique())

# ---- Step 2: Load admissions (for temporal ordering) ----
print("\nStep 2: Loading admissions...")
admissions = pd.read_csv(
    f'{MIMIC_PATH}/hosp/admissions.csv',
    parse_dates=['admittime', 'dischtime']
)
print(f"  Total admissions: {len(admissions):,}")

# Build stay → (subject_id, hadm_id, current_admittime) mapping
icustays = pd.read_csv(
    f'{MIMIC_PATH}/icu/icustays.csv',
    parse_dates=['intime', 'outtime']
)

stay_info = icustays[
    icustays['stay_id'].isin(cohort_stay_ids)
][['stay_id', 'hadm_id', 'subject_id']].drop_duplicates()

stay_info = stay_info.merge(
    admissions[['hadm_id', 'admittime']].rename(
        columns={'admittime': 'current_admittime'}
    ),
    on='hadm_id', how='left'
)

print(f"  Mapped {len(stay_info)} cohort stays to admissions")

# ---- Step 3: Load diagnoses ----
print("\nStep 3: Loading diagnoses_icd...")
diag_df = pd.read_csv(
    f'{MIMIC_PATH}/hosp/diagnoses_icd.csv',
    dtype={'icd_code': str}
)

# Filter to ICD-10 only
diag_df = diag_df[diag_df['icd_version'] == 10].copy()
print(f"  ICD-10 diagnoses: {len(diag_df):,}")

# Attach admittime and subject_id to each diagnosis
diag_df = diag_df.merge(
    admissions[['hadm_id', 'subject_id', 'admittime']],
    on='hadm_id', how='left'
)

# Filter to cohort patients only
cohort_subject_ids = set(stay_info['subject_id'].values)
diag_df = diag_df[diag_df['subject_id'].isin(cohort_subject_ids)]
print(f"  Filtered to cohort patients: {len(diag_df):,}")

# ---- Step 4: Prior-admission hierarchical propagation ----
print("\nStep 4: Hierarchical propagation (prior admissions only)...")

stay_code_sets = {}
prior_count = 0
no_prior_count = 0

for _, row in stay_info.iterrows():
    stay_id = row['stay_id']
    subj_id = row['subject_id']
    current_admit = row['current_admittime']

    prior_diags = diag_df[
        (diag_df['subject_id'] == subj_id) &
        (diag_df['admittime'] < current_admit)
    ]

    if prior_diags.empty:
        no_prior_count += 1
        continue

    prior_count += 1
    code_set = set()
    for code in prior_diags['icd_code'].values:
        code_set.update(hierarchical_propagate(code))
    stay_code_sets[stay_id] = code_set

print(f"  Stays with prior diagnoses: {prior_count}")
print(f"  First-time admissions (no prior): {no_prior_count}")

# ---- Step 5: Frequency-based feature selection ----
print("\nStep 5: Selecting features by frequency...")

long_rows = []
for stay_id, codes in stay_code_sets.items():
    for code in codes:
        long_rows.append({'stay_id': stay_id, 'code': code})

if not long_rows:
    print("  WARNING: No hierarchical codes generated. Saving empty features.")
    result = cohort[['stay_id']].copy()
    result.to_csv('data/icd_features.csv', index=False)
    json.dump([], open('data/icd_vocabulary.json', 'w'))
    exit(0)

code_long = pd.DataFrame(long_rows)

total_stays = len(cohort_stay_ids)
min_count = max(1, int(total_stays * MIN_FREQUENCY))
code_freq = (code_long.groupby('code')['stay_id']
             .nunique()
             .sort_values(ascending=False))
selected_codes = code_freq[code_freq >= min_count].head(MAX_FEATURES).index.tolist()

print(f"  Total unique hierarchical codes: {len(code_freq)}")
print(f"  Min count threshold: {min_count} ({MIN_FREQUENCY:.0%} of {total_stays} stays)")
print(f"  Selected: {len(selected_codes)} codes")

if selected_codes:
    print(f"  Top 10 most frequent:")
    for code in selected_codes[:10]:
        count = code_freq[code]
        print(f"    {code:10s}  {count:5d} stays ({count/total_stays:.1%})")

# ---- Step 6: Build binary feature matrix ----
print("\nStep 6: Building binary feature matrix...")

code_long = code_long[code_long['code'].isin(selected_codes)]
code_long['value'] = 1
icd_matrix = code_long.pivot_table(
    index='stay_id', columns='code', values='value',
    fill_value=0, aggfunc='max'
)
icd_matrix.columns = [f'icd_{c}' for c in icd_matrix.columns]
icd_matrix = icd_matrix.reset_index()

# Left join with full cohort (first-time patients get all zeros)
result = cohort[['stay_id']].merge(icd_matrix, on='stay_id', how='left')
icd_cols = [c for c in result.columns if c.startswith('icd_')]
result[icd_cols] = result[icd_cols].fillna(0).astype(int)

# ---- Step 7: Save ----
print("\nStep 7: Saving outputs...")

os.makedirs('data', exist_ok=True)
result.to_csv('data/icd_features.csv', index=False)

vocabulary = sorted(selected_codes)
with open('data/icd_vocabulary.json', 'w') as f:
    json.dump(vocabulary, f, indent=2)

print(f"  Saved: data/icd_features.csv  ({result.shape[0]} stays x {len(icd_cols)} features)")
print(f"  Saved: data/icd_vocabulary.json  ({len(vocabulary)} codes)")

# ---- Summary ----
print("\n" + "=" * 60)
print("ICD-10 FEATURE EXTRACTION COMPLETE!")
print("=" * 60)

n_with_codes = (result[icd_cols].sum(axis=1) > 0).sum()
print(f"\n  Cohort: {len(result)} stays")
print(f"  Stays with prior ICD history: {n_with_codes} ({n_with_codes/len(result):.1%})")
print(f"  Stays without (first-time): {len(result) - n_with_codes}")
print(f"  ICD feature columns: {len(icd_cols)}")

print("\nNext step:")
print("  Run: python create_sequences.py")
print("  or:  python create_sequences_sliding_window.py")
print("=" * 60)
