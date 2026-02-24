"""
Train bleeding prediction model
Implements the architecture from Park et al. (2022) PMC9672494
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

print("="*60)
print("Bleeding Prediction Model Training")
print("="*60)

# Load data
print("\nLoading processed data...")
data = np.load('processed_data.npz')

X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"\nData loaded:")
print(f"  Train: {X_train.shape}, Bleeding rate: {y_train.mean():.1%}")
print(f"  Val: {X_val.shape}, Bleeding rate: {y_val.mean():.1%}")
print(f"  Test: {X_test.shape}, Bleeding rate: {y_test.mean():.1%}")
print(f"  Features ({len(feature_names)}): {list(feature_names)}")

# Check for class imbalance
print(f"\nClass distribution:")
print(f"  Training: {(y_train==0).sum()} non-bleeding, {(y_train==1).sum()} bleeding")
print(f"  Class ratio: {(y_train==0).sum() / (y_train==1).sum():.1f}:1")

# Calculate class weights to handle imbalance
class_weight = {
    0: 1.0,
    1: (y_train == 0).sum() / (y_train == 1).sum()
}
print(f"  Using class weights: {class_weight}")

# Build model - GRU architecture (from Park et al.)
print("\n" + "="*60)
print("Building Model")
print("="*60)

model = Sequential([
    # Input layer with masking for missing values
    Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])),
    
    # First GRU layer (128 units)
    GRU(128, return_sequences=True, name='gru_1'),
    Dropout(0.3, name='dropout_1'),
    
    # Second GRU layer (64 units)
    GRU(64, name='gru_2'),
    Dropout(0.3, name='dropout_2'),
    
    # Dense layer
    Dense(32, activation='relu', name='dense_1'),
    Dropout(0.2, name='dropout_3'),
    
    # Output layer
    Dense(1, activation='sigmoid', name='output')
], name='Bleeding_Prediction_GRU')

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("\nModel Architecture:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Train model
print("\n" + "="*60)
print("Training Model")
print("="*60)
print("\nThis may take 30-60 minutes depending on your hardware...")
print("Training with early stopping (patience=10 epochs)")
print()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
print("\nPlotting training history...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# AUC
axes[0, 1].plot(history.history['auc'], label='Train AUC')
axes[0, 1].plot(history.history['val_auc'], label='Val AUC')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].set_title('Training and Validation AUC')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train Precision')
axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Training and Validation Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train Recall')
axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title('Training and Validation Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("  Saved to: training_history.png")

# Evaluate on test set
print("\n" + "="*60)
print("Test Set Evaluation")
print("="*60)

# Get predictions
y_pred_proba = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
auroc = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest Set Results:")
print(f"  AUROC: {auroc:.4f}")
print(f"  Target (from paper): 0.94-0.95")
print(f"  HAS-BLED baseline: 0.60-0.62")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, 
    target_names=['No Bleeding', 'Bleeding'],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 No    Yes")
print(f"Actual No     {cm[0,0]:5d} {cm[0,1]:5d}")
print(f"       Yes    {cm[1,0]:5d} {cm[1,1]:5d}")

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nClinical Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  PPV (Precision): {ppv:.4f}")
print(f"  NPV: {npv:.4f}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

# Add labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['No Bleeding', 'Bleeding'])
ax.set_yticklabels(['No Bleeding', 'Bleeding'])
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j],
                      ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                      fontsize=20)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n  Saved to: confusion_matrix.png")

# ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'GRU Model (AUC = {auroc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Bleeding Prediction', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("  Saved to: roc_curve.png")

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, linewidth=2, label=f'GRU Model (AP = {ap_score:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("  Saved to: precision_recall_curve.png")

# Save final model
print("\n" + "="*60)
print("Saving Model")
print("="*60)

model.save('bleeding_prediction_model.h5')
print("\nModel saved to: bleeding_prediction_model.h5")
print("  File size: ~60-80 MB")
print("  Format: TensorFlow/Keras HDF5")

# Save model summary to text file
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n" + "="*60 + "\n")
    f.write("Test Results\n")
    f.write("="*60 + "\n")
    f.write(f"AUROC: {auroc:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"Precision: {ppv:.4f}\n")
    f.write(f"NPV: {npv:.4f}\n")

print("  Model summary saved to: model_summary.txt")

# Summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

print(f"\nFinal Performance:")
print(f"  Test AUROC: {auroc:.4f}")
print(f"  Target from paper: 0.94-0.95")
print(f"  Improvement over HAS-BLED: {((auroc - 0.61) / 0.61 * 100):.1f}%")

print(f"\nFiles created:")
print(f"  ✓ bleeding_prediction_model.h5  - Trained model")
print(f"  ✓ best_model.h5                  - Best checkpoint")
print(f"  ✓ training_history.png           - Training curves")
print(f"  ✓ confusion_matrix.png           - Confusion matrix")
print(f"  ✓ roc_curve.png                  - ROC curve")
print(f"  ✓ precision_recall_curve.png     - PR curve")
print(f"  ✓ model_summary.txt              - Model details")

print("\n" + "="*60)
print("Next Steps:")
print("="*60)
print("1. Review training_history.png to check for overfitting")
print("2. Check if AUROC meets target (0.94)")
print("3. If AUROC is low, try:")
print("   - Increase sample size (use more patients)")
print("   - Add more features")
print("   - Try LSTM instead of GRU")
print("   - Adjust hyperparameters")
print("\n" + "="*60)