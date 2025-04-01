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
FREQ = 8  # Hz
LENGTH = 60 # seconds
OVERLAP = 75 # 75%, 50%, 25% overlap

# Set up pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def normalize(X):
    return np.clip((X - 50) / (100 - 50 + 1e-8), 0, 1)

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



# Check for GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

            
# Create filenames
train_file = f'datasets/ablation_study/train_freq{FREQ}_len{LENGTH}_overlap{OVERLAP}.feather'
test_file = f'datasets/ablation_study/test_freq{FREQ}_len{LENGTH}_overlap{OVERLAP}.feather'


# Load data
train = pd.read_feather(train_file)
test = pd.read_feather(test_file)
X_train = train
X_test = test

# Extract labels
y_train = X_train[["Label"]].values
y_test = X_test[["Label"]].values

# Convert features to numpy arrays
X_train = np.stack(X_train["Segment"].values)
X_test = np.stack(X_test["Segment"].values)


#Normalize data based on global values
X_train = normalize(X_train)
X_test = normalize(X_test)

#Oversampling based on duplication with added noise
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
model = keras.Sequential([
    layers.Conv1D(filters=16, kernel_size=9, strides=1, padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.001),),
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.001)),
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.L2(0.001)),
    layers.MaxPool1D(pool_size=2),
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(16, activation="relu",kernel_initializer='he_normal'),
    layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L2(0.01),kernel_initializer='glorot_uniform')
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

model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train.flatten()),
    y=y_train.flatten()
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)
# Train model with timing
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=128,
    validation_data=(X_test,y_test),
    callbacks=[early_stopping,reduce_lr],
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


model_path = f"models/{FREQ}-{LENGTH}.keras"
model.save(model_path)              