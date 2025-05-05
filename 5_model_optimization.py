import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc,classification_report, roc_auc_score
from sklearn.utils import shuffle
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
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
from sklearn.metrics import (roc_auc_score, precision_score, 
                           recall_score, accuracy_score, confusion_matrix)
import tempfile

#from keras import layers
#from keras import regularizers
# Define all combinations to test
FREQ = 1  # Hz
LENGTH = 60 # seconds
OVERLAP = 50 # 75%, 50%, 25% overlap
FINE_TUNE_EPOCHS = 4  # Epochs for fine-tuning after pruning
RESULTS_FILE = "pruning_quant_results.csv"

# Set up pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def evaluate_tflite_metrics(interpreter, x_test, y_test, threshold=0.5, convert=False):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    y_pred = []
    y_true = []
    
    if convert:
        input_scale, input_zero_point = input_details['quantization']
        output_scale, output_zero_point = output_details['quantization']
        print("Input scale/zero_point:", input_scale, input_zero_point)
        print("Output scale/zero_point:", output_scale, output_zero_point)
    
    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0)

        if convert:
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        else:
            input_data = input_data.astype(input_details['dtype'])

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details['index'])

        if convert:
            pred = (pred.astype(np.float32) - output_zero_point) * output_scale

        y_pred.append(pred[0][0])
        y_true.append(y_test[i][0])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate all metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'auc': roc_auc_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary),  # sensitivity
        'specificity': tn / (tn + fp + 1e-7),  # True Negative Rate
        'f1_score': 2 * tp / (2 * tp + fp + fn + 1e-7),  # direct calculation
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }
    
    return metrics


# Benchmark function for TFLite models
def benchmark_inference(interpreter, x_test, reps=100):
    input_details = interpreter.get_input_details()[0]
    
    # Warmup
    if input_details['dtype'] == np.float32:
        interpreter.set_tensor(input_details['index'], x_test[0:1].astype(np.float32))
    else:
        interpreter.set_tensor(input_details['index'], x_test[0:1].astype(np.uint8))
    interpreter.invoke()
    
    # Benchmark
    start = time.time()
    for _ in range(reps):
        if input_details['dtype'] == np.float32:
            interpreter.set_tensor(input_details['index'], x_test[0:1].astype(np.float32))
        else:
            interpreter.set_tensor(input_details['index'], x_test[0:1].astype(np.uint8))
        interpreter.invoke()
    return (time.time() - start) * 1000 / reps


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
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = keras.Sequential([
                    keras.layers.Reshape((1, X_train.shape[1], 1), input_shape=(X_train.shape[1], 1)),
                    # Replace Conv1D with Conv2D
                    keras.layers.Conv2D(filters=16, kernel_size=(1, 9), strides=(1, 1), 
                                    padding="same", activation="relu",
                                    kernel_initializer='he_normal'),
                    keras.layers.MaxPool2D(pool_size=(1, 2)),
                    keras.layers.Dropout(0.1),
                    keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding="same", 
                                    activation="relu", kernel_initializer='he_normal'),
                    keras.layers.MaxPool2D(pool_size=(1, 2)),
                    keras.layers.Dropout(0.1),
                    keras.layers.Conv2D(filters=64, kernel_size=(1, 3), padding="same", 
                                    activation="relu", kernel_initializer='he_normal'),
                    keras.layers.MaxPool2D(pool_size=(1, 2)),
                    keras.layers.Dropout(0.1),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(0.25),
                    keras.layers.Dense(16, activation="relu", kernel_initializer='he_normal'),
                    keras.layers.Dense(1, activation="sigmoid",
                                    kernel_initializer='glorot_uniform')
                ])
reduce_lr = keras.callbacks.ReduceLROnPlateau(
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
    patience=20,
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
    class_weight={0 : 1, 1 : 2},
    verbose=1
)
model_path = f"models/{FREQ}-{LENGTH}.keras"
model.save(model_path)  



######## Quantization of model#######################

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)
q_aware_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy",
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.F1Score(name="f1_score"),
                keras.metrics.Recall(name='recall')]
)

q_aware_model.summary()
history_q = q_aware_model.fit(
    X_train, y_train,
    epochs=4,
    batch_size=128,
    validation_data=(X_test,y_test),
    callbacks=[early_stopping,reduce_lr],
    class_weight={0 : 1, 1 : 2},
    verbose=1
)
# ----------------------------
# 3. Main Experiment Loop
# ----------------------------
results = []
print("Evaluating baseline model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()


def representative_dataset():
    for i in range(len(X_train)):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]


converter_post = tf.lite.TFLiteConverter.from_keras_model(model)
converter_post.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter_post.convert()
interpreter_post = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter_post.allocate_tensors()


converter_int= tf.lite.TFLiteConverter.from_keras_model(model)
converter_int.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int.inference_input_type = tf.uint8
converter_int.inference_output_type = tf.uint8
converter_int.representative_dataset = representative_dataset
int_tflite_model = converter_int.convert()
interpreter_int = tf.lite.Interpreter(model_content=int_tflite_model)
interpreter_int.allocate_tensors()

converter_int2= tf.lite.TFLiteConverter.from_keras_model(model)
converter_int2.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int2.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
#converter_int2.inference_input_type = tf.int16
#converter_int2.inference_output_type = tf.int16
converter_int2.representative_dataset = representative_dataset
int2_tflite_model = converter_int2.convert()
interpreter_int2 = tf.lite.Interpreter(model_content=int2_tflite_model)
interpreter_int2.allocate_tensors()


converter_float= tf.lite.TFLiteConverter.from_keras_model(model)
converter_float.optimizations = [tf.lite.Optimize.DEFAULT]
converter_float.target_spec.supported_types = [tf.float16]
float_tflite_model = converter_float.convert()
interpreter_float = tf.lite.Interpreter(model_content=float_tflite_model)
interpreter_float.allocate_tensors()



q_aware_converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
q_aware_converter.optimizations = [tf.lite.Optimize.DEFAULT]
q_aware_tflite_model = q_aware_converter.convert()
q_aware_interpreter = tf.lite.Interpreter(model_content=q_aware_tflite_model)
q_aware_interpreter.allocate_tensors()
print("baseline", evaluate_tflite_metrics(interpreter, X_test, y_test))
print("dynamic", evaluate_tflite_metrics(interpreter_post, X_test, y_test))

# Quantize manually
# Extract scale and zero point for input/output


print("full-int", evaluate_tflite_metrics(interpreter_int, X_test, y_test, convert=True))
print("float16", evaluate_tflite_metrics(interpreter_float, X_test, y_test))
print("Act16-weight8", evaluate_tflite_metrics(interpreter_int2, X_test, y_test))
print("q-aware", evaluate_tflite_metrics(q_aware_interpreter, X_test, y_test))



import pathlib

tflite_models_dir = pathlib.Path("./models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"baseline.tflite"
tflite_model_file.write_bytes(tflite_model)
# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"q-aware.tflite"
tflite_model_quant_file.write_bytes(q_aware_tflite_model)

tflite_model_int_file = tflite_models_dir/"int8.tflite"
tflite_model_int_file.write_bytes(int_tflite_model)

tflite_model_int2_file = tflite_models_dir/"w8-a16.tflite"
tflite_model_int2_file.write_bytes(int2_tflite_model)

# Save the quantized model:
tflite_model_post_file = tflite_models_dir/"dynamic.tflite"
tflite_model_post_file.write_bytes(quantized_tflite_model)


# Save the quantized model:
tflite_model_float_file = tflite_models_dir/"float16.tflite"
tflite_model_float_file.write_bytes(float_tflite_model)

print("Original model size:", os.path.getsize(tflite_model_file) / 1024, "KB")
print("Q-aware model size:", os.path.getsize(tflite_model_quant_file) / 1024, "KB")
print("Full integer model size:", os.path.getsize(tflite_model_int_file) / 1024, "KB")
print("Activation16 Weight8 size:", os.path.getsize(tflite_model_int2_file) / 1024, "KB")
print("Post quantization size:", os.path.getsize(tflite_model_post_file) / 1024, "KB")
print("Float16 quantization size:", os.path.getsize(tflite_model_float_file) / 1024, "KB")