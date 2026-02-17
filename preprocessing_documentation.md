# MIMIC-IV Bleeding Prediction Preprocessing Pipeline

## Overview
This preprocessing pipeline extracts and processes data from MIMIC-IV to predict bleeding risk in anticoagulated patients in the ICU. The pipeline follows best practices from recent bleeding prediction research and the MIMIC-IV data processing literature.

## Pipeline Architecture

```
Raw MIMIC-IV Data
    ↓
1. Cohort Selection
    ↓
2. Outcome Definition
    ↓
3. Feature Extraction
    ↓
4. Temporal Alignment
    ↓
5. Data Cleaning
    ↓
6. Missing Value Handling
    ↓
7. Normalization
    ↓
8. Train/Validation/Test Split
    ↓
Processed Dataset for Modeling
```

## Detailed Steps

### 1. Cohort Selection

**Objective**: Identify ICU patients on anticoagulation or antiplatelet therapy

**Inclusion Criteria**:
- ICU admission in MIMIC-IV
- Age ≥ 18 years
- Received anticoagulation or antiplatelet therapy during ICU stay
- Minimum ICU stay of 24 hours

**Anticoagulation Medications** (using `pharmacy` and `prescriptions` tables):
- Warfarin
- Heparin (unfractionated)
- Low molecular weight heparins (enoxaparin, dalteparin)
- Direct oral anticoagulants (dabigatran, rivaroxaban, apixaban, edoxaban)
- Antiplatelet agents (aspirin, clopidogrel, ticagrelor, prasugrel)

**SQL Tables Used**:
- `icu.icustays` - ICU admission records
- `hosp.admissions` - Hospital admission data
- `hosp.patients` - Patient demographics
- `hosp.prescriptions` - Medication orders
- `icu.inputevents` - IV medication administration
- `pharmacy.pharmacy` - Pharmacy dispenses

**Key Identifiers**:
- `subject_id` - Patient identifier
- `hadm_id` - Hospital admission identifier
- `stay_id` - ICU stay identifier

### 2. Outcome Definition

**Primary Outcome**: Bleeding event during ICU stay

**Bleeding Definition** (composite criteria):

**A. Transfusion-based**:
- Packed red blood cells (PRBC) transfusion
- From `icu.inputevents` table with:
  - `itemid` for blood products
  - Volume ≥ 1 unit

**B. ICD Code-based** (from `hosp.diagnoses_icd`):

ICD-10 Codes for Bleeding:
- **Gastrointestinal bleeding**:
  - K92.0 (Hematemesis)
  - K92.1 (Melena)
  - K92.2 (GI hemorrhage, unspecified)
  - K25.0-K25.6 (Gastric ulcer with hemorrhage)
  - K26.0-K26.6 (Duodenal ulcer with hemorrhage)
  - K27.0-K27.6 (Peptic ulcer with hemorrhage)

- **Intracranial hemorrhage**:
  - I60.x (Subarachnoid hemorrhage)
  - I61.x (Intracerebral hemorrhage)
  - I62.x (Other nontraumatic intracranial hemorrhage)
  - S06.4-S06.6 (Traumatic intracranial hemorrhage)

- **Urinary tract bleeding**:
  - R31.x (Hematuria)
  - N02.x (Recurrent and persistent hematuria)

- **Respiratory bleeding**:
  - R04.0 (Epistaxis)
  - R04.1 (Hemorrhage from throat)
  - R04.2 (Hemoptysis)

- **Other bleeding sites**:
  - D62 (Acute posthemorrhagic anemia)
  - R58 (Hemorrhage, not elsewhere classified)
  - T81.0 (Hemorrhage and hematoma complicating a procedure)

**Composite Bleeding Definition**:
```
Bleeding = (PRBC transfusion ≥ 1 unit) AND (ICD code for bleeding/hemorrhage control)
```

**Exclusion Criteria**:
- Bleeding events prior to anticoagulation start
- Trauma-related bleeding if trauma is primary diagnosis
- Bleeding events after ICU discharge

**Time Windows**:
- Observation window: 12 hours before bleeding event
- Prediction horizon: 3 hours ahead
- Total time frame: Drug start → Event or ICU discharge

### 3. Feature Extraction

**Total Features**: 27 variables organized into categories

#### Demographics (Static Features)
1. **Age** - Years at admission
2. **Sex** - Male/Female (binary)
3. **Race** - Categorical (White/Black/Asian/Hispanic/Other)
4. **Weight** - Kg (from chartevents)
5. **Height** - cm (from chartevents)

#### Vital Signs (Time-varying)
6. **Heart Rate** (HR) - beats/min
7. **Systolic Blood Pressure** (SBP) - mmHg
8. **Diastolic Blood Pressure** (DBP) - mmHg
9. **Mean Arterial Pressure** (MAP) - mmHg
10. **Respiratory Rate** (RR) - breaths/min
11. **Temperature** - °C
12. **SpO2** - Oxygen saturation %

#### Laboratory Values (Time-varying)
**Hematology**:
13. **Hemoglobin** (Hb) - g/dL
14. **Hematocrit** (Hct) - %
15. **Platelet count** - K/µL
16. **White blood cell count** (WBC) - K/µL
17. **International Normalized Ratio** (INR)
18. **Activated Partial Thromboplastin Time** (aPTT) - seconds
19. **Prothrombin Time** (PT) - seconds

**Chemistry**:
20. **Creatinine** - mg/dL
21. **Blood Urea Nitrogen** (BUN) - mg/dL
22. **Sodium** - mEq/L
23. **Potassium** - mEq/L
24. **Glucose** - mg/dL

**Other**:
25. **Lactate** - mmol/L
26. **pH** - Arterial blood gas
27. **Glasgow Coma Scale** (GCS) - Score 3-15

#### Derived Features
- **Shock Index** = HR / SBP
- **Pulse Pressure** = SBP - DBP
- **Anion Gap** = Na - (Cl + HCO3)

### 4. Temporal Alignment

**Temporal Structure**:
- **Observation Window**: 12 hours prior to prediction point
- **Time Resolution**: 30-minute intervals
- **Sequence Length**: 24 time steps (12 hours / 30 min)

**Alignment Strategy**:
1. Define time zero (T0) as:
   - For bleeding cases: 3 hours before bleeding event
   - For controls: Random time during ICU stay (≥15 hours after admission)

2. Create observation window: [T0 - 12 hours, T0]

3. Divide observation window into 30-minute bins

4. Aggregate measurements within each bin:
   - Vitals: Mean value
   - Labs: Last value (if multiple)
   - Static features: Replicate across all time steps

**SQL Tables for Temporal Data**:
- `icu.chartevents` - Vital signs and GCS
- `hosp.labevents` - Laboratory measurements
- Time-stamped using `charttime` or `storetime`

### 5. Data Cleaning

**Outlier Detection and Handling**:

For each continuous feature:
1. **Logical Range Checks**:
   ```python
   VALID_RANGES = {
       'heart_rate': (0, 300),
       'sbp': (0, 300),
       'dbp': (0, 200),
       'temperature': (25, 45),  # Celsius
       'spo2': (0, 100),
       'hemoglobin': (0, 25),
       'platelet': (0, 2000),
       'creatinine': (0, 25),
       'gcs': (3, 15)
   }
   ```
   Values outside logical ranges → Set to NaN

2. **Statistical Outlier Detection**:
   - Remove values beyond 99th percentile
   - Cap at 99th percentile value rather than removing
   ```python
   p99 = feature.quantile(0.99)
   feature = feature.clip(upper=p99)
   ```

3. **Implausible Value Patterns**:
   - Remove repeated identical values (>6 consecutive)
   - Flag sudden jumps (>3 standard deviations from previous)

**Duplicate Removal**:
- Remove exact duplicate rows within same time bin
- Keep most recent measurement if multiple per bin

### 6. Missing Value Handling

**Missing Data Strategy** (hierarchical approach):

**Level 1: Forward Fill (within patient)**
- Carry forward last valid observation
- Maximum forward fill: 2 hours (4 time steps)
```python
df_patient.fillna(method='ffill', limit=4)
```

**Level 2: Linear Interpolation**
- For gaps ≤ 2 hours after forward fill
```python
df_patient.interpolate(method='linear', limit=4)
```

**Level 3: Population Mean/Median Imputation**
- Use cohort statistics for remaining missingness
- Vitals: Median
- Labs: Median (by age group and sex)

**Level 4: Masking**
- Create binary mask indicating missing values
- Pass to model for informed handling
```python
mask = ~df.isna()  # True where data exists
```

**Missing Data Documentation**:
Track missingness per feature:
```python
missing_report = {
    'feature_name': {
        'percent_missing': float,
        'temporal_pattern': 'random/systematic',
        'imputation_method': 'forward_fill/interpolation/median'
    }
}
```

### 7. Normalization and Standardization

**Feature Scaling Strategy**:

**Continuous Features** (z-score normalization):
```python
# Fit on training set only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Formula: z = (x - μ) / σ
```

**Categorical Features** (one-hot encoding):
```python
# Sex, Race
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(categorical_features)
```

**Normalization by Feature Type**:
- **Vitals**: Z-score (population mean/std)
- **Labs**: Z-score (age-sex adjusted if available)
- **GCS**: Min-max scaling [0,1] (ordinal scale)
- **Binary**: Keep as 0/1

**Preserve Temporal Structure**:
```python
# For each patient sequence
for patient in patients:
    # Shape: (n_timesteps, n_features)
    # Normalize across features, preserve time dimension
    patient_scaled = scaler.transform(patient)
```

### 8. Train/Validation/Test Split

**Split Strategy** (patient-level):

**Option A: Temporal Split** (recommended for deployment):
```
Training:   2008-2015 (60% of patients)
Validation: 2015-2017 (20% of patients)
Testing:    2017-2019 (20% of patients)
```

**Option B: Random Split** (for initial development):
```
Training:   60% of patients
Validation: 20% of patients  
Testing:    20% of patients
```

**Critical Requirements**:
1. **Patient-level splitting**: All ICU stays for one patient in same set
   ```python
   from sklearn.model_selection import GroupShuffleSplit
   
   splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
   train_idx, test_idx = next(splitter.split(X, y, groups=subject_ids))
   ```

2. **Stratification**: Maintain bleeding event proportion
   ```python
   # Target: ~20% bleeding events in each split
   train_bleeding_rate = y_train.mean()  # Should ≈ 0.20
   ```

3. **Class Balance**:
   - Original: ~20% bleeding cases (1,134/5,670)
   - Consider SMOTE or class weights for imbalanced data

**Cross-Validation** (for hyperparameter tuning):
```python
from sklearn.model_selection import GroupKFold

# 5-fold CV on training set
kfold = GroupKFold(n_splits=5)
for train_idx, val_idx in kfold.split(X_train, y_train, groups=subject_ids_train):
    # Fit and evaluate model
    pass
```

## Data Output Format

### For RNN Models (Sequential)

**Shape**: `(n_patients, n_timesteps, n_features)`

```python
# Example output
X_train.shape = (3402, 24, 27)  # 3402 patients, 24 time steps, 27 features
y_train.shape = (3402,)          # Binary outcome
masks_train.shape = (3402, 24, 27)  # Missing value masks
```

**Saved Format**:
```python
import numpy as np

np.savez_compressed('processed_data.npz',
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    masks_train=masks_train,
    masks_val=masks_val,
    masks_test=masks_test,
    feature_names=feature_names,
    scaler_params={'mean': scaler.mean_, 'std': scaler.scale_}
)
```

### For Gradient Boosting (Aggregated Features)

**Shape**: `(n_patients, n_aggregated_features)`

**Aggregation Strategy**:
```python
# For each patient's 12-hour window
aggregated_features = {
    'mean': features.mean(axis=0),        # Mean over time
    'std': features.std(axis=0),          # Std dev over time
    'min': features.min(axis=0),          # Minimum value
    'max': features.max(axis=0),          # Maximum value
    'slope': calculate_slope(features),   # Linear trend
    'last': features[-1],                 # Most recent value
}
```

**Example**: 27 base features → 27 × 6 = 162 aggregated features

**Saved Format**:
```python
import pandas as pd

df = pd.DataFrame({
    'subject_id': subject_ids,
    'bleeding': y,
    **{f'hr_mean': hr_means, 
       f'hr_std': hr_stds,
       # ... all aggregated features
    }
})

df.to_csv('processed_data_xgboost.csv', index=False)
```

## Quality Checks

**Pre-Processing Validation**:
1. ✓ No negative ages
2. ✓ Vital signs within physiological ranges
3. ✓ No duplicate patient-timepoint combinations
4. ✓ All anticoagulation start times < bleeding event times
5. ✓ Bleeding events occur during ICU stay

**Post-Processing Validation**:
1. ✓ No missing values in final dataset (or properly masked)
2. ✓ Features properly normalized (mean≈0, std≈1 for training)
3. ✓ No data leakage (test patients not in train)
4. ✓ Class balance maintained across splits
5. ✓ Temporal ordering preserved in sequences

**Data Quality Metrics**:
```python
def quality_report(df):
    return {
        'n_patients': df['subject_id'].nunique(),
        'n_sequences': len(df),
        'bleeding_rate': df['bleeding'].mean(),
        'missing_rate': df.isna().mean().mean(),
        'feature_correlations': df.corr()['bleeding'].sort_values(),
        'temporal_coverage': check_temporal_coverage(df)
    }
```

## Expected Sample Sizes

Based on MIMIC-IV hemorrhage prediction study:

| Dataset | Total | Bleeding | Non-Bleeding | Bleeding % |
|---------|-------|----------|--------------|------------|
| **Training** | 3,402 | 680 | 2,722 | 20.0% |
| **Validation** | 1,134 | 227 | 907 | 20.0% |
| **Testing** | 1,134 | 227 | 907 | 20.0% |
| **Total** | 5,670 | 1,134 | 4,536 | 20.0% |

## Performance Benchmarks

Target preprocessing should enable:
- **Basic RNN (11 features)**: AUROC ≥ 0.61
- **1-Layer GRU (18 features)**: AUROC ≥ 0.95
- **2-Layer LSTM (27 features)**: AUROC ≥ 0.94

## Dependencies

```bash
# Database access
pip install psycopg2-binary sqlalchemy

# Data processing
pip install pandas numpy scipy scikit-learn

# Visualization
pip install matplotlib seaborn

# MIMIC-specific
pip install wfdb  # For waveform data if needed
```

## References

1. Park et al. (2022). Machine Learning Model for the Prediction of Hemorrhage in ICU. J Pers Med. PMC9672494.
2. Gupta et al. (2022). An Extensive Data Processing Pipeline for MIMIC-IV. ML4H Proceedings.
3. Johnson et al. (2023). MIMIC-IV, a freely accessible EHR dataset. Scientific Data.
4. Harutyunyan et al. (2019). Multitask learning and benchmarking with clinical time series data. Scientific Data.

## Next Steps

1. Implement cohort selection SQL queries
2. Build feature extraction pipeline
3. Create temporal alignment module
4. Develop data cleaning utilities
5. Implement missing value handlers
6. Build normalization pipeline
7. Create train/test splitting logic
8. Validate preprocessing with known benchmarks
