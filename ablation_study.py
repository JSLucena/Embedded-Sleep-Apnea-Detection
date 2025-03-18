import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc,classification_report, roc_auc_score
from sklearn.utils import shuffle
from keras import regularizers
from sklearn.model_selection import train_test_split
import os
import gc
import time
import sys
from sklearn.utils import class_weight
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# Define all combinations to test
TARGET_FREQUENCIES = [4, 8, 16]  # Hz
SEGMENT_LENGTHS = [20, 30]  # seconds
WINDOW_OVERLAPS = [0.5]  # 75%, 50%, 25% overlap


test = {
    20: (4,5),
    25: (5,5),
    30: (6,5),
    80: (10,8),
    100: (10,10),
    120: (12,10),
    160: (16,10),
    200: (20,10),
    240: (16,15),
    320: (20,16),
    400: (20,20),
    480: (24,20)
}



# Set up pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Function to get model size in MB
def get_model_size(model):
    # Save model to a temporary file
    model.save('temp_model.h5')
    # Get file size in MB
    size_bytes = os.path.getsize('temp_model.h5')
    size_mb = size_bytes / (1024 * 1024)
    # Remove temporary file
    os.remove('temp_model.h5')
    return size_mb

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
        return -keras.backend.sum(alpha * keras.backend.pow(1. - pt, gamma) * keras.backend.log(pt))
    return focal_loss_fn

# Create a results DataFrame
results_df = pd.DataFrame(columns=[
    'frequency', 'segment_length', 'overlap', 'trainable_params', 'model_size_mb',
    'training_time', 'accuracy', 'auc', 'recall', 'precision', 'f1_score'
])

# Check for GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

# Iterate through all combinations
for freq in TARGET_FREQUENCIES:
    for length in SEGMENT_LENGTHS:
        for overlap in WINDOW_OVERLAPS:
            # Convert overlap to percentage for filename
            overlap_pct = int(overlap * 100)
            
            # Create filenames
            train_file = f'datasets/ablation_study/train_freq{freq}_len{length}_overlap{overlap_pct}.feather'
            test_file = f'datasets/ablation_study/test_freq{freq}_len{length}_overlap{overlap_pct}.feather'
            
            # Check if files exist
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"Skipping combination - files not found: freq={freq}, len={length}, overlap={overlap_pct}")
                continue
                
            print(f"\n{'='*80}")
            print(f"Processing: freq={freq}Hz, segment_length={length}s, overlap={overlap_pct}%")
            print(f"{'='*80}")
            
            # Load data
            try:
                # Load data
                train = pd.read_feather(train_file)
                test = pd.read_feather(test_file)

                # Split into train and test
                combined = pd.concat([train, test], axis=0, ignore_index=True)
                combined = shuffle(combined, random_state=42)
                X_train, X_test = train_test_split(combined, test_size=0.2, random_state=42, stratify=combined["Label"])

                # Extract labels
                y_train = X_train[["Label"]].values
                y_test = X_test[["Label"]].values

                # Convert features to numpy arrays
                X_train = np.stack(X_train["Segment"].values)
                X_test = np.stack(X_test["Segment"].values)

                # Reshape for Keras
                X_train = X_train.reshape(-1, X_train.shape[1], 1)
                X_test = X_test.reshape(-1, X_test.shape[1], 1)

                # Scaling data
                scaler = StandardScaler()
                X_train= scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
                X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

                # Store the segment length for proper reshaping later
                
                segment_length = X_train.shape[1]
                # Apply SMOTE on the scaled data
                #smote = SMOTE(random_state=42)
                #X_train_resampled, y_train_resampled = smote.fit_resample(
                #    X_train,
                #    y_train.flatten()
                #)

                X_train = X_train.reshape(-1, segment_length, 1)
                y_train = y_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, segment_length, 1)

                model = keras.Sequential([
                    layers.BatchNormalization(input_shape=(X_train.shape[1], 1)),
                    layers.Conv1D(filters=5, kernel_size=3, strides=1, padding="valid", activation="relu"),
                    layers.MaxPooling1D(pool_size=2),
                    layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
                    layers.MaxPooling1D(pool_size=2),
                    layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
                    layers.MaxPooling1D(pool_size=2),
                    layers.BatchNormalization(),
                    layers.Flatten(),
                    
                    #layers.Dropout(0.25),
                    layers.Dense(16, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
                    layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L2(0.01))
                ])
                
                # Set up optimizer and compile
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=10000,
                    decay_rate=0.9
                )
                opt = keras.optimizers.Adam(learning_rate=lr_schedule,clipvalue=1.0)
                model.compile(
                    optimizer=opt,
                    loss="binary_crossentropy",
                    metrics=["accuracy",
                             keras.metrics.AUC(name='auc'),
                             keras.metrics.Precision(name='precision'),
                             keras.metrics.F1Score(name="f1_score"),
                             keras.metrics.Recall(name='recall')]
                )
                # Get model summary
                model.summary()

                # Get model size in MB
                model_size = get_model_size(model)
                print(f"Model size: {model_size:.2f} MB")
                
                # Prepare callbacks
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
                # Calculate class weights
                class_weights = class_weight.compute_class_weight(
                    'balanced',
                    classes=np.unique(y_train.flatten()),
                    y=y_train.flatten()
                )
                class_weights = dict(enumerate(class_weights))

                # Print class weights
                print("Class weights:", class_weights)
                # Train model with timing
                start_time = time.time()
                history = model.fit(
                    X_train, y_train,
                    epochs=200,  # Reduced from 200 for faster iteration
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    class_weight=class_weights,
                    verbose=1
                )
                training_time = time.time() - start_time
                print(f"Training time: {training_time:.2f} seconds")
                
                # Evaluate on test set
                results = model.evaluate(X_test, y_test, return_dict=True)
                
                # Print results
                print(f"Test Loss: {results['loss']:.4f}")
                print(f"Test Accuracy: {results['accuracy']:.4f}")
                print(f"Test AUC: {results['auc']:.4f}")
                print(f"Test Recall: {results['recall']:.4f}")
                print(f"Test Precision: {results['precision']:.4f}")
                print(f"Test F1-score: {results['f1_score']:.4f}")
                
                # Add results to DataFrame
                new_row = {
                    'frequency': freq,
                    'segment_length': length,
                    'overlap': overlap,
                    'model_size_mb': model_size,
                    'training_time': training_time,
                    'accuracy': results['accuracy'],
                    'auc': results['auc'],
                    'recall': results['recall'],
                    'precision': results['precision'],
                    'f1_score': results['f1_score']
                }
                
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Save results after each iteration
                results_df.to_csv('ablation_study_results.csv', index=False)
                
                # Optional: Generate and save plots for each model
                if True:  # Set to True if you want to save plots
                    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
                    
                    # Plot accuracy
                    axs[0, 0].plot(history.history["accuracy"], label="Train")
                    axs[0, 0].plot(history.history["val_accuracy"], label="Val")
                    axs[0, 0].set_title("Accuracy")
                    axs[0, 0].legend()
                    
                    # Plot loss
                    axs[0, 1].plot(history.history["loss"], label="Train")
                    axs[0, 1].plot(history.history["val_loss"], label="Val")
                    axs[0, 1].set_title("Loss")
                    axs[0, 1].legend()
                    
                    # Plot AUC
                    axs[0, 2].plot(history.history["auc"], label="Train")
                    axs[0, 2].plot(history.history["val_auc"], label="Val")
                    axs[0, 2].set_title("AUC")
                    axs[0, 2].legend()
                    
                    # Plot Precision/Recall
                    axs[1, 0].plot(history.history["precision"], label="Train Precision")
                    axs[1, 0].plot(history.history["val_precision"], label="Val Precision")
                    axs[1, 0].plot(history.history["recall"], label="Train Recall")
                    axs[1, 0].plot(history.history["val_recall"], label="Val Recall")
                    axs[1, 0].set_title("Precision and Recall")
                    axs[1, 0].legend()
                    
                    # Confusion Matrix
                    y_pred = (model.predict(X_test) > 0.5).astype("int32")
                    cm = confusion_matrix(y_test, y_pred)
                    axs[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    axs[1, 1].set_title('Confusion Matrix')
                    axs[1, 1].set_xlabel('Predicted')
                    axs[1, 1].set_ylabel('True')
                    
                    # ROC Curve
                    y_pred_prob = model.predict(X_test)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    axs[1, 2].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                    axs[1, 2].plot([0, 1], [0, 1], 'k--')
                    axs[1, 2].set_title('ROC Curve')
                    axs[1, 2].legend()
                    axs[1, 2].set_xlabel('False Positive Rate', fontsize=12)
                    axs[1, 2].set_ylabel('True Positive Rate', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(f'images/ablation_model_freq{freq}_len{length}_overlap{overlap_pct}.png')
                    plt.close()
                
                
                # Evaluate on the test set
                y_pred = model.predict(X_test)
                y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to binary predictions

                # Classification report
                print("Classification Report:")
                print(classification_report(y_test, y_pred_classes, target_names=["Normal (0)", "Abnormal (1)"]))

                # Confusion matrix
                print("Confusion Matrix:")
                print(confusion_matrix(y_test, y_pred_classes))

                # AUC-ROC score
                auc_score = roc_auc_score(y_test, y_pred)
                print(f"AUC-ROC Score: {auc_score:.4f}")
                # Clean up to prevent memory issues
                del model, X_train, X_test, y_train, y_test
                gc.collect()
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"Error processing combination freq={freq}, len={length}, overlap={overlap_pct}")
                print(f"Error: {str(e)}")
                continue

# Final save of results
results_df.to_csv('ablation_study_results.csv', index=False)

# Display summary of best models
print("\nTop 5 models by AUC:")
print(results_df.sort_values(by='auc', ascending=False).head(5)[['frequency', 'segment_length', 'overlap', 'auc', 'accuracy', 'recall', 'precision']])

print("\nTop 5 models by Recall:")
print(results_df.sort_values(by='recall', ascending=False).head(5)[['frequency', 'segment_length', 'overlap', 'auc', 'accuracy', 'recall', 'precision']])

print("\nTop 5 models by Precision:")
print(results_df.sort_values(by='precision', ascending=False).head(5)[['frequency', 'segment_length', 'overlap', 'auc', 'accuracy', 'recall', 'precision']])

# Create visualization of results
plt.figure(figsize=(15, 10))

# Plot AUC by frequency, segment length, and overlap
plt.subplot(2, 2, 1)
for freq in TARGET_FREQUENCIES:
    subset = results_df[results_df['frequency'] == freq]
    for overlap in WINDOW_OVERLAPS:
        overlap_subset = subset[subset['overlap'] == overlap]
        if not overlap_subset.empty:
            plt.plot(overlap_subset['segment_length'], overlap_subset['auc'], 
                     marker='o', label=f'Freq={freq}, Overlap={overlap}')
plt.xlabel('Segment Length (seconds)')
plt.ylabel('AUC')
plt.title('AUC by Segment Length')
plt.grid(True)
plt.legend()

# Plot Recall by segment length
plt.subplot(2, 2, 2)
for freq in TARGET_FREQUENCIES:
    subset = results_df[results_df['frequency'] == freq]
    for overlap in WINDOW_OVERLAPS:
        overlap_subset = subset[subset['overlap'] == overlap]
        if not overlap_subset.empty:
            plt.plot(overlap_subset['segment_length'], overlap_subset['recall'], 
                     marker='o', label=f'Freq={freq}, Overlap={overlap}')
plt.xlabel('Segment Length (seconds)')
plt.ylabel('Recall')
plt.title('Recall by Segment Length')
plt.grid(True)
plt.legend()

# Plot accuracy vs model size
plt.subplot(2, 2, 3)
plt.scatter(results_df['model_size_mb'], results_df['accuracy'], alpha=0.7)
for i, row in results_df.iterrows():
    plt.annotate(f"{row['frequency']}Hz, {row['segment_length']}s", 
                 (row['model_size_mb'], row['accuracy']), 
                 fontsize=8)
plt.xlabel('Model Size (MB)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Model Size')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'images/ablation_overall.png')
plt.close()