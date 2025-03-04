import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

train = 'datasets/trainset-segments.feather'
test = 'datasets/testset-segments.feather'
pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

train = pd.read_feather(train)  
test = pd.read_feather(test)


X_train = np.stack(train["Segment"].values)  # Convert to NumPy array
X_test = np.stack(test["Segment"].values)

y_train = train[["Label"]].values  # For Binary Classification (Apnea vs. Normal)
y_test = test[["Label"]].values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

X_train = X_train / 100.0  # Normalize SpO₂ (assuming max 100%)
X_test = X_test / 100.0
print(tf.config.list_physical_devices('GPU'))

# Model based on the
model = keras.Sequential([

    layers.BatchNormalization(input_shape=(X_train.shape[1], 1)),
    #layers.Conv1D(filters=16, kernel_size=8, strides=2, padding="same", activation="relu", input_shape=(X_train.shape[1], 1)),
    layers.Conv1D(filters=6, kernel_size=25, strides=1, padding="same", activation="relu"),
    layers.MaxPooling1D(pool_size=2, strides=1),
    layers.Conv1D(filters=50, kernel_size=10, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=2, strides=1),
    layers.Conv1D(filters=30, kernel_size=15, activation="relu", padding="same"),
    layers.MaxPooling1D(pool_size=5, strides=1),
    layers.BatchNormalization(),
    layers.Flatten(),
    #layers.Dense(64, activation="relu"),
    #layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
# Reshape X to match Keras expectations (batch_size, timesteps, features)
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Evaluate on Test Set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot Training History
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Training Progress")
plt.show()

