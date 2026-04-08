"""
Create temporal sequences from extracted features
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("="*60)
print("Creating Temporal Sequences")
print("="*60)

# Configuration
OBSERVATION_WINDOW_HOURS = 12
PREDICTION_HORIZON_HOURS = 3
TIME_RESOLUTION_MINUTES = 30

# Load data
print("\nLoading data...")
cohort = pd.read_csv('data/cohort.csv', parse_dates=['intime', 'outtime', 'anticoag_start', 'bleeding_time'])
vitals = pd.read_csv('data/vitals_sample.csv', parse_dates=['charttime'])
labs = pd.read_csv('data/labs_sample.csv', parse_dates=['charttime'])

print(f"  Cohort: {len(cohort)} patients")
print(f"  Vitals: {len(vitals):,} measurements")
print(f"  Labs: {len(labs):,} measurements")

# Load ICD features (optional — produced by extract_icd_features.py)
icd_features = None
icd_feature_names = []
try:
    icd_df = pd.read_csv('data/icd_features.csv')
    icd_feature_names = [c for c in icd_df.columns if c.startswith('icd_')]
    if icd_feature_names:
        icd_features = icd_df.set_index('stay_id')
        print(f"  ICD features: {len(icd_feature_names)} codes loaded")
    else:
        print("  ICD features: file found but no icd_* columns")
except FileNotFoundError:
    print("  ICD features: not found (run extract_icd_features.py first)")

# Combine features
print("\nCombining features...")
vitals_subset = vitals[['stay_id', 'charttime', 'feature_name', 'valuenum']]
labs_subset = labs[['stay_id', 'charttime', 'feature_name', 'valuenum']]
all_features = pd.concat([vitals_subset, labs_subset], ignore_index=True)

print(f"  Total measurements: {len(all_features):,}")

# Create sequences
print("\nCreating temporal sequences...")

n_timesteps = int(OBSERVATION_WINDOW_HOURS * 60 / TIME_RESOLUTION_MINUTES)
print(f"  Timesteps per sequence: {n_timesteps}")

sequences = []

for idx, patient in cohort.iterrows():
    if idx % 100 == 0:
        print(f"  Processing {idx}/{len(cohort)}...")
    
    stay_id = patient['stay_id']
    
    # Define T0
    if patient['bleeding'] == 1 and pd.notna(patient['bleeding_time']):
        t0 = patient['bleeding_time'] - timedelta(hours=PREDICTION_HORIZON_HOURS)
    else:
        stay_duration_hours = patient['los_hours']
        min_hours = OBSERVATION_WINDOW_HOURS + PREDICTION_HORIZON_HOURS
        
        if stay_duration_hours < min_hours:
            continue
        
        random_offset = np.random.uniform(min_hours, stay_duration_hours)
        t0 = patient['intime'] + timedelta(hours=random_offset)
    
    # Observation window
    window_start = t0 - timedelta(hours=OBSERVATION_WINDOW_HOURS)
    window_end = t0
    
    if window_start < patient['anticoag_start']:
        continue
    
    # Extract features
    patient_features = all_features[
        (all_features['stay_id'] == stay_id) &
        (all_features['charttime'] >= window_start) &
        (all_features['charttime'] <= window_end)
    ].copy()
    
    if len(patient_features) == 0:
        continue
    
    # Create time bins
    time_bins = pd.date_range(
        start=window_start,
        end=window_end,
        periods=n_timesteps + 1
    )
    
    patient_features['time_bin'] = pd.cut(
        patient_features['charttime'],
        bins=time_bins,
        labels=range(n_timesteps),
        include_lowest=True
    )
    
    # Aggregate
    sequence = patient_features.groupby(['time_bin', 'feature_name'])['valuenum'].mean().unstack()
    
    # Reindex to ensure all timesteps
    sequence = sequence.reindex(range(n_timesteps))
    
    sequences.append({
        'stay_id': stay_id,
        'sequence': sequence.values,
        'feature_names': sequence.columns.tolist(),
        'bleeding': patient['bleeding'],
        'age': patient['age']
    })

print(f"\nCreated {len(sequences)} sequences")

# Convert to arrays
print("\nConverting to arrays...")

# First, collect all unique feature names across all sequences
all_feature_names = set()
for seq in sequences:
    all_feature_names.update(seq['feature_names'])

feature_names = sorted(list(all_feature_names))
n_features = len(feature_names)

print(f"  Total unique features: {n_features}")
print(f"  Features: {feature_names}")

X = np.zeros((len(sequences), n_timesteps, n_features))
y = np.zeros(len(sequences))

for i, seq in enumerate(sequences):
    # Create a mapping from feature names to indices
    for j, feature in enumerate(feature_names):
        if feature in seq['feature_names']:
            # Find the column index in the sequence
            feature_idx = seq['feature_names'].index(feature)
            X[i, :, j] = seq['sequence'][:, feature_idx]
        else:
            # Feature not present for this patient - will be NaN (filled with 0s by default)
            X[i, :, j] = np.nan
    
    y[i] = seq['bleeding']

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Missing rate: {np.isnan(X).mean():.1%}")

# Build static ICD feature array (aligned with sequences)
if icd_features is not None and icd_feature_names:
    n_icd = len(icd_feature_names)
    X_icd = np.zeros((len(sequences), n_icd))
    for i, seq in enumerate(sequences):
        sid = seq['stay_id']
        if sid in icd_features.index:
            X_icd[i] = icd_features.loc[sid, icd_feature_names].values
    print(f"  X_icd shape: {X_icd.shape}")
else:
    X_icd = None

# Handle missing values
print("\nHandling missing values...")
from sklearn.impute import SimpleImputer

X_2d = X.reshape(-1, n_features)
imputer = SimpleImputer(strategy='median')
X_2d_imputed = imputer.fit_transform(X_2d)
X_imputed = X_2d_imputed.reshape(X.shape)

print(f"  Missing rate after imputation: {np.isnan(X_imputed).mean():.1%}")

# Split data
print("\nSplitting data...")
from sklearn.model_selection import train_test_split

indices = np.arange(len(X_imputed))
train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

X_train = X_imputed[train_idx]
X_val = X_imputed[val_idx]
X_test = X_imputed[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

if X_icd is not None:
    X_icd_train = X_icd[train_idx]
    X_icd_val = X_icd[val_idx]
    X_icd_test = X_icd[test_idx]

print(f"  Train: {len(X_train)} ({y_train.mean():.1%} bleeding)")
print(f"  Val: {len(X_val)} ({y_val.mean():.1%} bleeding)")
print(f"  Test: {len(X_test)} ({y_test.mean():.1%} bleeding)")

# Normalize
print("\nNormalizing...")
from sklearn.preprocessing import StandardScaler

X_train_2d = X_train.reshape(-1, n_features)
scaler = StandardScaler()
scaler.fit(X_train_2d)

X_train_norm = scaler.transform(X_train_2d).reshape(X_train.shape)
X_val_norm = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
X_test_norm = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

# Save
print("\nSaving processed data...")
save_dict = dict(
    X_train=X_train_norm,
    X_val=X_val_norm,
    X_test=X_test_norm,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    feature_names=feature_names,
)
if X_icd is not None:
    save_dict['X_icd_train'] = X_icd_train
    save_dict['X_icd_val'] = X_icd_val
    save_dict['X_icd_test'] = X_icd_test
    save_dict['icd_feature_names'] = np.array(icd_feature_names)
    print(f"  Including ICD features: {X_icd_train.shape[1]} codes")

np.savez_compressed('processed_data.npz', **save_dict)

print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print("\nProcessed data saved to: processed_data.npz")
print(f"  X_train: {X_train_norm.shape}")
print(f"  X_val: {X_val_norm.shape}")
print(f"  X_test: {X_test_norm.shape}")

print("\nNext step:")
print("  Run: python train_model.py")
print("="*60)
