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
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# Define all combinations to test
TARGET_FREQUENCIES = [1,4, 8, 16]  # Hz
SEGMENT_LENGTHS = [11,15,20,30,60]  # seconds
WINDOW_OVERLAPS = [0.75]  # 75%, 50%, 25% overlap


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

def moving_average(data, window_size=5):
    """Applies a moving average filter to each time series in the dataset."""
    kernel = np.ones(window_size) / window_size  # Define averaging kernel
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=data)  
    return smoothed  # Keeps shape (samples, timesteps)

def normalize(X):
    return np.clip((X - 50) / (100 - 50 + 1e-8), 0, 1)

def normalize2(X):
    # Reshape to (num_samples, timesteps) if needed
    original_shape = X.shape
    X_reshaped = X.reshape(len(X), -1)  # Shape: (N, timesteps)
    
    # Compute min/max per segment (keepdims for broadcasting)
    mins = np.min(X_reshaped, axis=1, keepdims=True)
    maxs = np.max(X_reshaped, axis=1, keepdims=True)
    
    # Normalize per segment
    X_normalized = (X_reshaped - mins) / (maxs - mins + 1e-8)
    
    return X_normalized.reshape(original_shape)

def simple_duplication_oversampling(X, y, minority_class=1, duplication_factor=2, noise_std=0.005, noise_max=0.01):

    # Find indices of minority class
    minority_indices = np.where(y.flatten() == minority_class)[0]

    # Duplicate the minority samples
    duplication_indices = np.tile(minority_indices, duplication_factor)

    # Apply small Gaussian noise
    noise = np.random.normal(loc=0, scale=noise_std, size=X[duplication_indices].shape)
    
    # Limit max noise range (avoid extreme outliers)
    noise = np.clip(noise, -noise_max, noise_max)

    # Add noise and ensure values remain in [0,1]
    X_augmented = np.clip(X[duplication_indices], 0, 1)

    # Combine original data with augmented minority samples
    X_resampled = np.vstack([X, X_augmented])
    y_resampled = np.concatenate([y.flatten(), np.full(len(X_augmented), minority_class)])

    return X_resampled, y_resampled


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
                #combined = pd.concat([train, test], axis=0, ignore_index=True)
                #combined = normalize(combined)

                #combined = shuffle(combined, random_state=42)
                X_train = train
                X_test = test
                #X_train, X_test = train_test_split(combined, test_size=0.1, random_state=42, stratify=combined["Label"])
                #X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=42, stratify=X_train["Label"])
                #X_test = test
                # Extract labels
                y_train = X_train[["Label"]].values
                #y_val = X_val[["Label"]].values
                y_test = X_test[["Label"]].values

                # Convert features to numpy arrays
                X_train = np.stack(X_train["Segment"].values)
                #X_val = np.stack(X_val["Segment"].values)
                X_test = np.stack(X_test["Segment"].values)

                # Reshape for Keras
                #X_train = X_train.reshape(-1, X_train.shape[1], 1)
                #X_val = X_val.reshape(-1, X_val.shape[1], 1)
                #X_test = X_test.reshape(-1, X_test.shape[1], 1)


                X_train = normalize(X_train)
                X_test = normalize(X_test)
                
                # Apply Moving Average (Window Size = 5)
                #X_train = moving_average(X_train, window_size=freq*3)
                #X_val = moving_average(X_val, window_size=freq*3)
                #X_test = moving_average(X_test, window_size=freq*3)


                # Store the segment length for proper reshaping later
                
                #segment_length = X_train.shape[1]
                # Apply SMOTE on the scaled data
                #smote = SMOTE(random_state=42)
                #X_train_resampled, y_train_resampled = smote.fit_resample(
                #    X_train,
                #    y_train.flatten()
                #)

                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_train_resampled, y_train_resampled = simple_duplication_oversampling(
                    X_train_flat, 
                    y_train, 
                    minority_class=1, 
                    duplication_factor=3
                )
                X_train = X_train_resampled
                y_train = y_train_resampled
                X_train = X_train_resampled.reshape(-1, X_train_resampled.shape[1], 1)
                y_train = y_train_resampled.reshape(-1, 1)
                X_test = X_test.reshape(-1, X_test.shape[1], 1)
                #X_val = X_val.reshape(-1, X_val.shape[1], 1)

                model = keras.Sequential([
                    layers.Reshape((1, X_train.shape[1], 1), input_shape=(X_train.shape[1], 1)),
                    # Replace Conv1D with Conv2D
                    layers.Conv2D(filters=16, kernel_size=(1, 9), strides=(1, 1), 
                                    padding="same", activation="relu",
                                    kernel_initializer='he_normal'),
                    layers.MaxPool2D(pool_size=(1, 2)),
                    layers.Dropout(0.1),
                    layers.Conv2D(filters=32, kernel_size=(1, 5), padding="same", 
                                    activation="relu", kernel_initializer='he_normal'),
                    layers.MaxPool2D(pool_size=(1, 2)),
                    layers.Dropout(0.1),
                    layers.Conv2D(filters=64, kernel_size=(1, 3), padding="same", 
                                    activation="relu", kernel_initializer='he_normal'),
                    layers.MaxPool2D(pool_size=(1, 2)),
                    layers.Dropout(0.1),
                    layers.Flatten(),
                    layers.Dropout(0.25),
                    layers.Dense(16, activation="relu", kernel_initializer='he_normal'),
                    layers.Dense(1, activation="sigmoid",
                                    kernel_initializer='glorot_uniform')
                ])
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=1e-5
                )
                opt = keras.optimizers.Adam(learning_rate=0.001,clipvalue=1.0)
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
                #model_size = get_model_size(model)
                #print(f"Model size: {model_size:.2f} MB")
                
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
                    epochs=300,  # Reduced from 200 for faster iteration
                    batch_size=128,
                    validation_data=(X_test,y_test),
                    callbacks=[early_stopping,reduce_lr],
                    class_weight={0 : 1, 1 : 2},
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
                
                trainable_count = model.count_params()
                #print(trainable_count)
                # Add results to DataFrame
                new_row = {
                    'frequency': freq,
                    'segment_length': length,
                    'overlap': overlap,
                #    'model_size_mb': model_size,
                    'training_time': training_time,
                    'trainable_parameters' : trainable_count,
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

                    # Plot confusion matrix with values
                    im = axs[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

                    # Add text annotations
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            axs[1, 1].text(j, i, f'{cm[i, j]}', 
                                        ha='center', 
                                        va='center', 
                                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

                    axs[1, 1].set_title('Confusion Matrix')
                    axs[1, 1].set_xlabel('Predicted')
                    axs[1, 1].set_ylabel('True')

                    # Optional: Add axis labels
                    axs[1, 1].set_xticks([0, 1])
                    axs[1, 1].set_yticks([0, 1])
                    axs[1, 1].set_xticklabels(['Negative', 'Positive'])
                    axs[1, 1].set_yticklabels(['Negative', 'Positive'])
                    
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
                    plt.savefig(f'images/ablation_model_freq{freq}_len{length}_overlap{overlap_pct}.pdf')
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

                raw_predictions = model.predict(X_test)
                print("Prediction distribution:", np.histogram(y_pred, bins=10))
                print("Mean prediction value:", np.mean(y_pred))
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
plt.scatter(results_df['trainable_parameters'], results_df['accuracy'], alpha=0.7)
for i, row in results_df.iterrows():
    plt.annotate(f"{row['frequency']}Hz, {row['segment_length']}s", 
                 (row['trainable_parameters'], row['accuracy']), 
                 fontsize=8)
plt.xlabel('Model Size (trainable parameters)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Model Size')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'images/ablation_overall.pdf')
plt.close()