import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.utils import shuffle
from keras import regularizers
from sklearn.model_selection import train_test_split
train = 'datasets/trainset-segments.feather'
test = 'datasets/testset-segments.feather'
pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

train = pd.read_feather(train)  
test = pd.read_feather(test)

combined = pd.concat([train, test], axis=0, ignore_index=True)
combined = shuffle(combined, random_state=42)

X_train, X_test = train_test_split(
    combined, test_size=0.2, random_state=42  # Adjust test_size as needed
)
print(X_train.columns)
y_train = X_train[["Label"]].values  # For Binary Classification (Apnea vs. Normal)
y_test = X_test[["Label"]].values
X_train = np.stack(X_train["Segment"].values)  # Convert to NumPy array



X_test = np.stack(X_test["Segment"].values)


print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

X_train = X_train / 100.0  # Normalize SpO₂ (assuming max 100%)
X_test = X_test / 100.0

print(tf.config.list_physical_devices('GPU'))

# Model based on the
model = keras.Sequential([

    layers.BatchNormalization(input_shape=(X_train.shape[1], 1)),
  # First conv block - more filters
    layers.Conv1D(filters=6, kernel_size=25, strides=2, padding="same", activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    #layers.Dropout(0.2),
    # Second conv block
    layers.Conv1D(filters=50, kernel_size=10, padding="same", activation="relu",kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling1D(pool_size=2),
    #layers.Dropout(0.2),
    # Third conv block
    layers.Conv1D(filters=30, kernel_size=15, padding="same", activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    #layers.Dropout(0.2),
    layers.BatchNormalization(), 
    # Add LSTM layer to capture temporal dependencies
    #layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
    layers.Flatten(),
    # Fully connected layers
    #layers.Dense(128, activation="relu"),
    #layers.Dropout(0.5),
    layers.Dense(16, activation="relu",kernel_regularizer=regularizers.l2(0.01)),
    #layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name='auc'),  # Additional metrics
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    

model.summary()
# Reshape X to match Keras expectations (batch_size, timesteps, features)
X_train, y_train = shuffle(X_train, y_train, random_state=42)


X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)


# Train Model
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
class_weights = {0: 1.0, 1: 1.0}
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_test, y_test), callbacks=[early_stopping], class_weight=class_weights)

raw_predictions = model.predict(X_test)
print("Prediction distribution:", np.histogram(raw_predictions, bins=10))
print("Mean prediction value:", np.mean(raw_predictions))
# Evaluate on Test Set
results = model.evaluate(X_test, y_test, return_dict=True)
print(f"Test Loss: {results['loss']:.4f}")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test AUC: {results['auc']:.4f}")
print(f"Test Recall: {results['recall']:.4f}")
print(f"Test Precision: {results['precision']:.4f}")

# Set up the figure size and style
plt.figure(figsize=(20, 15))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Accuracy
plt.subplot(2, 3, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(loc="lower right")
plt.title("Training vs Validation Accuracy", fontsize=14)
plt.grid(True)

# Plot 2: Loss
plt.subplot(2, 3, 2)
plt.plot(history.history["loss"], label="Train Loss", color='red', linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", color='orange', linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc="upper right")
plt.title("Training vs Validation Loss", fontsize=14)
plt.grid(True)

# Plot 3: AUC
plt.subplot(2, 3, 3)
plt.plot(history.history["auc"], label="Train AUC", color='green', linewidth=2)
plt.plot(history.history["val_auc"], label="Val AUC", color='lime', linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("AUC", fontsize=12)
plt.legend(loc="lower right")
plt.title("Training vs Validation AUC", fontsize=14)
plt.grid(True)

# Plot 4: Precision/Recall
plt.subplot(2, 3, 4)
plt.plot(history.history["precision"], label="Train Precision", color='purple', linewidth=2)
plt.plot(history.history["val_precision"], label="Val Precision", color='magenta', linewidth=2)
plt.plot(history.history["recall"], label="Train Recall", color='blue', linewidth=2, linestyle='--')
plt.plot(history.history["val_recall"], label="Val Recall", color='cyan', linewidth=2, linestyle='--')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.legend(loc="lower right")
plt.title("Precision and Recall", fontsize=14)
plt.grid(True)

# Plot 5: Confusion Matrix
plt.subplot(2, 3, 5)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Apnea'], rotation=45)
plt.yticks(tick_marks, ['Normal', 'Apnea'])

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Plot 6: ROC Curve
plt.subplot(2, 3, 6)
y_pred_prob = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()

"""
# Learning Curves - Combined plot showing both accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", color='blue', linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", color='skyblue', linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(loc="lower right")
plt.title("Accuracy Curves", fontsize=14)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", color='red', linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", color='salmon', linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc="upper right")
plt.title("Loss Curves", fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()

# Precision-Recall Curve (more detailed)
plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

"""