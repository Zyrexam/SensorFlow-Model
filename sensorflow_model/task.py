import os
import numpy as np
import json
from tensorflow import keras
from keras import layers, models, regularizers
from sensorflow_model.dataset import get_dataset, normalize_features
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# #  CNN + LSTM
# def load_model():
#     model = keras.Sequential([
#         keras.Input(shape=(200, 12)),  # 2s window with 12 features

#         # 1st Conv Block
#         layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.2),

#         # 2nd Conv Block
#         layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.3),

#         # Optional deeper feature extraction
#         layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.MaxPooling1D(pool_size=2),
#         layers.Dropout(0.3),

#         # LSTM Layer
#         layers.LSTM(128, return_sequences=False),
#         layers.Dropout(0.4),

#         # Fully connected
#         layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
#         layers.Dropout(0.5),

#         # Output layer for 12 activity classes
#         layers.Dense(12, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model


def load_model():
    model = keras.Sequential([
        keras.Input(shape=(20, 12)), 

        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        # BiLSTM Layer
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.4),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),

        layers.Dense(12, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

def get_parameters(model):
    return model.get_weights()


def set_parameters(model, parameters):
    model.set_weights(parameters)


def load_data(partition_id, data_folder="myData"):
    client_folder = os.path.join(data_folder, f"Client_{partition_id + 1}")
    all_files = sorted([f for f in os.listdir(client_folder) if f.endswith(".csv")])
    
    if not all_files:
        raise ValueError(f"No CSV files found in folder {client_folder}")
    
    X_total, y_total = [], []

    for file in all_files:
        filepath = os.path.join(client_folder, file)
        X, y = get_dataset(filepath, window_size=20, stride=5, normalize=False)  # no normalization yet
        X_total.append(X)
        y_total.append(y)

    X = np.concatenate(X_total, axis=0)
    y = np.concatenate(y_total, axis=0)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize training set and save stats once
    x_train, scaler = normalize_features(x_train, save=True)

    # Apply same stats to test set
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    return x_train, y_train, x_test, y_test



def export_to_tflite(model, output_path="model.tflite"):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,       
        tf.lite.OpsSet.SELECT_TF_OPS          
    ]

    converter._experimental_lower_tensor_list_ops = False  
    converter.experimental_enable_resource_variables = True 

    converter.optimizations = {tf.lite.Optimize.DEFAULT} 

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model) # type: ignore




def export_feature_stats(x_train, save_path="feature_stats.json"):
    # Compute mean and std per feature (across time and samples)
    feature_means = x_train.mean(axis=(0, 1)).tolist()
    feature_stds = x_train.std(axis=(0, 1)).tolist()
    
    stats = {"mean": feature_means, "std": feature_stds}
    
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"[✓] Feature stats saved to: {save_path}")
