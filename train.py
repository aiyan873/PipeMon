# =====================
#  Colab Setup
# =====================
!pip install xgboost lightgbm tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from google.colab import files
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import pickle

# =====================
#  Step 1: Upload & Load Processed Data
# =====================
print("Please upload your processed pipe vibration dataset CSV file")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"Loading dataset: {filename}")

df = pd.read_csv(filename)

print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nTarget distribution:")
print(df['target'].value_counts().sort_index())

print(f"\nFirst 5 rows:")
print(df.head())

# =====================
# Step 2: Prepare Features and Target
# =====================
feature_columns = [col for col in df.columns if col not in ['timestamp', 'target']]
X = df[feature_columns]
y = df['target']

print(f"\nFeature columns ({len(feature_columns)}):")
print(feature_columns)

print(f"\nMissing values per column:")
print(X.isnull().sum().sum())

print(f"\nDataset statistics:")
print(f"Total samples: {len(df)}")
print(f"Number of features: {len(feature_columns)}")
target_names = {0: 'Normal', 1: 'Leak', 2: 'Burst'}
for target in [0, 1, 2]:
    count = len(df[df['target'] == target])
    percentage = (count / len(df)) * 100
    print(f"  Class {target} ({target_names[target]}): {count} samples ({percentage:.1f}%)")

# =====================
# Step 3: Train/Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =====================
# Step 4: Model Training (No Evaluation Yet)
# =====================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1)
}

trained_pipelines = {}

print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

# =====================
# Step 5: Feature Importance
# =====================
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors = ['skyblue', 'lightgreen', 'lightcoral']

for idx, (name, model) in enumerate(models.items()):
    pipe = trained_pipelines[name]

    if hasattr(pipe.named_steps['clf'], 'feature_importances_'):
        importances = pipe.named_steps['clf'].feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        top_features = feature_importance_df.head(10)
        axes[idx].barh(range(len(top_features)), top_features['importance'], color=colors[idx], alpha=0.7)
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['feature'])
        axes[idx].set_xlabel('Feature Importance')
        axes[idx].set_title(f'{name}\nTop 10 Features')
        axes[idx].grid(True, alpha=0.3, axis='x')

        print(f"\nTop 5 features for {name}:")
        for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
            print(f"{i}. {row['feature']}: {row['importance']:.4f}")

plt.tight_layout()
plt.show()

# =====================
# Step 6: TensorFlow Lite Conversion with Distillation for High Accuracy
# =====================
print("\n" + "="*50)
print("TENSORFLOW LITE CONVERSION WITH KNOWLEDGE DISTILLATION")
print("="*50)

def distill_to_neural_network(pipeline, name, X_train, y_train, X_test, y_test):
    scaler = pipeline.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get soft probabilities from the tree model
    y_train_prob = pipeline.predict_proba(X_train)
    y_test_prob = pipeline.predict_proba(X_test)

    # High-level NN architecture for better approximation
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train_scaled, y_train_prob,
        epochs=100, batch_size=32,
        validation_data=(X_test_scaled, y_test_prob),
        callbacks=[early_stop],
        verbose=0
    )

    return model, scaler

def convert_to_tflite(keras_model, X_test_scaled):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        for i in range(min(100, len(X_test_scaled))):
            yield [X_test_scaled[i:i+1].astype(np.float32)]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return converter.convert()

def test_tflite_model(tflite_model, X_test_scaled, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for i in range(len(X_test_scaled)):
        interpreter.set_tensor(input_details[0]['index'], X_test_scaled[i:i+1].astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data[0]))

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
    rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    return acc, prec, rec, f1, cm, predictions

tflite_results = {}
metrics_summary = []
confusion_matrices = {}
model_sizes = {}

for name, pipeline in trained_pipelines.items():
    print(f"\nDistilling and converting {name} to TFLite...")

    nn_model, scaler = distill_to_neural_network(pipeline, name, X_train, y_train, X_test, y_test)

    X_test_scaled = scaler.transform(X_test)
    tflite_model = convert_to_tflite(nn_model, X_test_scaled)
    tflite_acc, tflite_prec, tflite_rec, tflite_f1, tflite_cm, tflite_pred = test_tflite_model(tflite_model, X_test_scaled, y_test)

    tflite_results[name] = {
        "acc": tflite_acc,
        "prec": tflite_prec,
        "rec": tflite_rec,
        "f1": tflite_f1,
        "cm": tflite_cm,
        "predictions": tflite_pred
    }

    metrics_summary.append([f"{name} TFLite", tflite_acc, tflite_prec, tflite_rec, tflite_f1])
    confusion_matrices[f"{name} TFLite"] = tflite_cm

    # Save TFLite model
    tflite_filename = f'pipe_leak_{name.lower().replace(" ", "_")}_model.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

    # Get sizes
    with tempfile.NamedTemporaryFile() as tmp:
        pickle.dump(pipeline, tmp)
        tmp.flush()
        original_size = os.path.getsize(tmp.name) / 1024

    keras_model_path = f'temp_keras_{name}.h5'
    nn_model.save(keras_model_path)
    keras_size = os.path.getsize(keras_model_path) / 1024
    os.remove(keras_model_path)

    tflite_size = len(tflite_model) / 1024

    model_sizes[name] = {
        "original": original_size,
        "nn": keras_size,
        "tflite": tflite_size
    }

    print(f"\n===== {name} TFLite Results =====")
    print(f"Accuracy: {tflite_acc:.4f}")
    print(f"Precision: {tflite_prec:.4f}")
    print(f"Recall: {tflite_rec:.4f}")
    print(f"F1-Score: {tflite_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, tflite_pred, target_names=['Normal', 'Leak', 'Burst']))

# =====================
# Step 7: Comparison Table for TFLite Models
# =====================
metrics_df = pd.DataFrame(metrics_summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n" + "="*60)
print("TFLITE MODEL COMPARISON TABLE")
print("="*60)
print(metrics_df.to_string(index=False, float_format='%.4f'))
metrics_df.to_csv("tflite_model_comparison.csv", index=False)

# =====================
# Step 8: Confusion Matrix Visualization for TFLite Models
# =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_names = ['Normal', 'Leak', 'Burst']

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
    axes[idx].set_title(f'{name}\nAccuracy: {metrics_summary[idx][1]:.4f}', fontsize=12, fontweight='bold')
    axes[idx].grid(False)

plt.tight_layout()
plt.suptitle('Confusion Matrix Comparison - TFLite Pipe Leak Detection Models',
             fontsize=16, fontweight='bold', y=1.05)
plt.show()

# =====================
# Step 9: Model Accuracy Comparison (Bar Chart) for TFLite
# =====================
plt.figure(figsize=(8, 6))
model_names = metrics_df['Model']
accuracies = metrics_df['Accuracy']
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = plt.bar(model_names, accuracies, color=colors, alpha=0.7)
plt.ylabel("Accuracy", fontweight='bold')
plt.title("TFLite Model Accuracy Comparison", fontweight='bold')
plt.ylim(0, 1.1)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# =====================
# Step 10: Best TFLite Model Summary
# =====================
best_model_idx = np.argmax(metrics_df['Accuracy'])
best_model = metrics_df.iloc[best_model_idx]['Model']
best_accuracy = metrics_df.iloc[best_model_idx]['Accuracy']

print("\n" + "="*50)
print("BEST TFLITE MODEL SUMMARY")
print("="*50)
print(f"Best performing TFLite model: {best_model}")
print(f"Best accuracy: {best_accuracy:.4f}")
print("\nComplete metrics for best model:")
best_metrics = metrics_df.iloc[best_model_idx]
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    print(f"{metric}: {best_metrics[metric]:.4f}")

# =====================
# Step 11: Model Size & Accuracy Comparison
# =====================
print("\n" + "="*50)
print("MODEL SIZE & ACCURACY COMPARISON")
print("="*50)

print(f"{'Model':<20} {'Original Size (KB)':<20} {'NN Size (KB)':<15} {'TFLite Size (KB)':<18} {'TFLite Accuracy':<15}")
for name in trained_pipelines.keys():
    sizes = model_sizes[name]
    acc = tflite_results[name]['acc']
    print(f"{name:<20} {sizes['original']:<20.2f} {sizes['nn']:<15.2f} {sizes['tflite']:<18.2f} {acc:<15.4f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
models_list = list(trained_pipelines.keys())
original_sizes = [model_sizes[name]['original'] for name in models_list]
nn_sizes = [model_sizes[name]['nn'] for name in models_list]
tflite_sizes = [model_sizes[name]['tflite'] for name in models_list]
accuracies = [tflite_results[name]['acc'] for name in models_list]

# Size comparison (stacked for original, nn, tflite)
ax1.bar(models_list, original_sizes, color='lightcoral', alpha=0.7, label='Original')
ax1.bar(models_list, nn_sizes, bottom=original_sizes, color='lightblue', alpha=0.7, label='NN')
ax1.bar(models_list, tflite_sizes, bottom=[o + n for o, n in zip(original_sizes, nn_sizes)], color='lightgreen', alpha=0.7, label='TFLite')
ax1.set_ylabel('Model Size (KB)', fontweight='bold')
ax1.set_title('Model Size Comparison', fontweight='bold')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

bars2 = ax2.bar(models_list, accuracies, color=['lightcoral','lightblue','lightgreen'], alpha=0.7)
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_title('TFLite Model Accuracy Comparison', fontweight='bold')
ax2.set_ylim(0.6, 1.0)
for bar, acc in zip(bars2, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Original vs Neural Network vs TensorFlow Lite',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
