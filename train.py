# =====================
#  Colab Setup
# =====================
!pip install xgboost lightgbm tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
# Step 4: Model Training, Cross-Validation & Evaluation
# =====================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1)
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
metrics_summary = []
confusion_matrices = {}
trained_pipelines = {}

print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="accuracy")

    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results[name] = {"cv_scores": cv_scores, "test_acc": acc}
    metrics_summary.append([name, acc, prec, rec, f1])
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} Results =====")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Leak', 'Burst']))

# =====================
# Step 5: Comparison Table
# =====================
metrics_df = pd.DataFrame(metrics_summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)
print(metrics_df.to_string(index=False, float_format='%.4f'))
metrics_df.to_csv("model_comparison.csv", index=False)

# =====================
# Step 6: Confusion Matrix Visualization
# =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_names = ['Normal', 'Leak', 'Burst']

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
    axes[idx].set_title(f'{name}\nAccuracy: {metrics_summary[idx][1]:.4f}', fontsize=12, fontweight='bold')
    axes[idx].grid(False)

plt.tight_layout()
plt.suptitle('Confusion Matrix Comparison - Pipe Leak Detection Models',
             fontsize=16, fontweight='bold', y=1.05)
plt.show()

# =====================
# Step 7: Model Accuracy Comparison (Bar Chart)
# =====================
plt.figure(figsize=(8, 6))
model_names = metrics_df['Model']
accuracies = metrics_df['Accuracy']
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = plt.bar(model_names, accuracies, color=colors, alpha=0.7)
plt.ylabel("Accuracy", fontweight='bold')
plt.title("Model Accuracy Comparison", fontweight='bold')
plt.ylim(0, 1.1)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

#plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# =====================
# Step 8: Best Model Summary
# =====================
best_model_idx = np.argmax(metrics_df['Accuracy'])
best_model = metrics_df.iloc[best_model_idx]['Model']
best_accuracy = metrics_df.iloc[best_model_idx]['Accuracy']

print("\n" + "="*50)
print("BEST MODEL SUMMARY")
print("="*50)
print(f"Best performing model: {best_model}")
print(f"Best accuracy: {best_accuracy:.4f}")
print("\nComplete metrics for best model:")
best_metrics = metrics_df.iloc[best_model_idx]
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    print(f"{metric}: {best_metrics[metric]:.4f}")

# =====================
# Step 9: Feature Importance
# =====================
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (name, model) in enumerate(models.items()):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)

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
# Step 10: TensorFlow Lite Conversion
# =====================
print("\n" + "="*50)
print("TENSORFLOW LITE CONVERSION")
print("="*50)

def create_neural_network_from_rf(rf_pipeline, X_train, y_train, X_test, y_test):
    scaler = rf_pipeline.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
              validation_data=(X_test_scaled, y_test), verbose=0)
    
    return model, scaler

def convert_to_tflite(keras_model, X_test_scaled):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset():
        for i in range(min(100, len(X_test_scaled))):
            yield [X_test_scaled[i:i+1].astype(np.float32)]
    converter.representative_dataset = representative_dataset
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
    
    return accuracy_score(y_test, predictions), predictions

if "Random Forest" in trained_pipelines:
    rf_pipeline = trained_pipelines["Random Forest"]
    nn_model, scaler = create_neural_network_from_rf(rf_pipeline, X_train, y_train, X_test, y_test)
    
    X_test_scaled = scaler.transform(X_test)
    tflite_model = convert_to_tflite(nn_model, X_test_scaled)
    tflite_accuracy, tflite_predictions = test_tflite_model(tflite_model, X_test_scaled, y_test)
    
    with open('pipe_leak_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    import pickle, tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        pickle.dump(rf_pipeline, tmp)
        tmp.flush()
        rf_size = os.path.getsize(tmp.name) / 1024
    
    keras_model_path = 'temp_keras_model.h5'
    nn_model.save(keras_model_path)
    keras_size = os.path.getsize(keras_model_path) / 1024
    os.remove(keras_model_path)
    
    tflite_size = len(tflite_model) / 1024
    
    rf_accuracy = rf_pipeline.score(X_test, y_test)
    nn_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    
    print(f"\n===== MODEL SIZE & ACCURACY COMPARISON =====")
    print(f"{'Model':<20} {'Size (KB)':<12} {'Accuracy':<10}")
    print(f"{'Random Forest':<20} {rf_size:<12.2f} {rf_accuracy:<10.4f}")
    print(f"{'Neural Network':<20} {keras_size:<12.2f} {nn_accuracy:<10.4f}")
    print(f"{'TensorFlow Lite':<20} {tflite_size:<12.2f} {tflite_accuracy:<10.4f}")
    
    # Visualization fixed here
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    models_list = ['Random Forest', 'Neural Network', 'TensorFlow Lite']
    sizes = [rf_size, keras_size, tflite_size]
    accuracies = [rf_accuracy, nn_accuracy, tflite_accuracy]
    
    bars1 = ax1.bar(models_list, sizes, color=['lightcoral','lightblue','lightgreen'], alpha=0.7)
    ax1.set_ylabel('Model Size (KB)', fontweight='bold')
    ax1.set_title('Model Size Comparison', fontweight='bold')
    ax1.set_yscale('log')
    for bar, size in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                 f'{size:.2f} KB', ha='center', va='bottom', fontweight='bold')
    
    bars2 = ax2.bar(models_list, accuracies, color=['lightcoral','lightblue','lightgreen'], alpha=0.7)
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Model Accuracy Comparison', fontweight='bold')
    ax2.set_ylim(0.6, 1.0)
    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Random Forest vs Neural Network vs TensorFlow Lite',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

else:
    print("Random Forest model not found for TFLite conversion!")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
