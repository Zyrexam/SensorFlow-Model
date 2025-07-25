
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers
from sensorflow_model.dataset import get_dataset, normalize_features
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



# # class GatedSensorFusion(layers.Layer):
# #     def __init__(self, conv_channels):
# #         super(GatedSensorFusion, self).__init__()
# #         self.conv_channels = conv_channels
# #         self.gate_dense = layers.Dense(conv_channels, activation='sigmoid')
# #         self.proj = layers.Dense(conv_channels)

# #     def call(self, x1, x2):
# #         # x1, x2: shape (batch, seq_len, conv_channels)
# #         concat = tf.concat([x1, x2], axis=-1)  # (batch, seq_len, 2 * conv_channels)
# #         gate = self.gate_dense(concat)         # (batch, seq_len, conv_channels)
# #         fused = gate * x1 + (1 - gate) * x2    # Gated fusion
# #         return self.proj(fused)                # Projected fused output

# # def load_model():
# #     model = keras.Sequential([
# #         keras.Input(shape=(20, 12)), 

# #         layers.Conv1D(64, 5, activation='relu', padding='same'),
# #         layers.BatchNormalization(),
# #         layers.MaxPooling1D(2),
# #         layers.Dropout(0.2),

# #         layers.Conv1D(128, 3, activation='relu', padding='same'),
# #         layers.BatchNormalization(),
# #         layers.MaxPooling1D(2),
# #         layers.Dropout(0.3),

# #         layers.Conv1D(256, 3, activation='relu', padding='same'),
# #         layers.BatchNormalization(),
# #         layers.MaxPooling1D(2),
# #         layers.Dropout(0.3),

# #         # BiLSTM Layer
# #         layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
# #         layers.Dropout(0.4),

# #         layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
# #         layers.Dropout(0.5),

# #         layers.Dense(12, activation='softmax')
# #     ])

# #     model.compile(
# #         optimizer='adam',
# #         loss='sparse_categorical_crossentropy',
# #         metrics=['accuracy'])
# #     return model

# # def get_parameters(model):
# #     return model.get_weights()


# # def set_parameters(model, parameters):
# #     model.set_weights(parameters)


# # def load_data(partition_id, data_folder="myData"):
# #     client_folder = os.path.join(data_folder, f"Client_{partition_id + 1}")
# #     all_files = sorted([f for f in os.listdir(client_folder) if f.endswith(".csv")])
    
# #     if not all_files:
# #         raise ValueError(f"No CSV files found in folder {client_folder}")
    
# #     X_total, y_total = [], []

# #     for file in all_files:
# #         filepath = os.path.join(client_folder, file)
# #         X, y = get_dataset(filepath, window_size=20, stride=5, normalize=False)  # no normalization yet
# #         X_total.append(X)
# #         y_total.append(y)

# #     X = np.concatenate(X_total, axis=0)
# #     y = np.concatenate(y_total, axis=0)

# #     # Split
# #     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Normalize training set and save stats once
# #     x_train, scaler = normalize_features(x_train, save=True)

# #     # Apply same stats to test set
# #     x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

# #     return x_train, y_train, x_test, y_test

class GatedSensorFusion(layers.Layer):
    def __init__(self, conv_channels):
        super(GatedSensorFusion, self).__init__()
        self.gate_dense = layers.Dense(conv_channels, activation='sigmoid')
        self.proj = layers.Dense(conv_channels)

    def call(self, x1, x2):
        concat = tf.concat([x1, x2], axis=-1)  # shape: (batch, time, 2 * channels)
        gate = self.gate_dense(concat)
        fused = gate * x1 + (1 - gate) * x2
        return self.proj(fused)


# # clusterFL
# # def load_model():
# #     # Two inputs: watch and earable
# #     input_watch = keras.Input(shape=(20, 6), name='watch_input')
# #     input_ear = keras.Input(shape=(20, 6), name='earable_input')

# #     # Shared Conv1D feature extractor
# #     def conv_block(x):
# #         x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.MaxPooling1D(2)(x)
# #         x = layers.Dropout(0.2)(x)

# #         x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.MaxPooling1D(2)(x)
# #         x = layers.Dropout(0.3)(x)

# #         x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.MaxPooling1D(2)(x)
# #         x = layers.Dropout(0.3)(x)
# #         return x

# #     # Apply conv block
# #     x_watch = conv_block(input_watch)
# #     x_ear = conv_block(input_ear)

# #     # Concatenate features
# #     fused = layers.Concatenate()([x_watch, x_ear])

# #     # Flatten
# #     flat = layers.Flatten()(fused)

# #     # FC1 layer (match old model: 300 units + relu)
# #     fc1 = layers.Dense(300, activation='relu')(flat)

# #     # Dropout (match old keep_prob=0.5)
# #     fc1 = layers.Dropout(0.5)(fc1)

# #     # FC2 → logits
# #     logits = layers.Dense(12, activation=None)(fc1)

# #     # Softmax output
# #     output = layers.Activation('softmax')(logits)

# #     model = keras.Model(inputs=[input_watch, input_ear], outputs=output)

# #     # Compile
# #     model.compile(
# #         optimizer='adam',
# #         loss='sparse_categorical_crossentropy',
# #         metrics=['accuracy']
# #     )

# #     return model



# cnn+bilstm

# def load_model():
#     # Two inputs: one for each sensor modality
#     input_watch = keras.Input(shape=(20, 6), name='watch_input')
#     input_ear = keras.Input(shape=(20, 6), name='earable_input')

#     # Shared Conv blocks for both branches
#     def conv_block(x):
#         x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling1D(2)(x)
#         x = layers.Dropout(0.2)(x)

#         x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling1D(2)(x)
#         x = layers.Dropout(0.3)(x)

#         x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling1D(2)(x)
#         x = layers.Dropout(0.3)(x)
#         return x

#     # Feature extraction
#     x_watch = conv_block(input_watch)
#     x_ear = conv_block(input_ear)

#     # Gated Fusion
#     fused = GatedSensorFusion(conv_channels=256)(x_watch, x_ear)

#     # BiLSTM and classification head
#     x = layers.Bidirectional(layers.LSTM(64))(fused)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
#     x = layers.Dropout(0.5)(x)
#     output = layers.Dense(12, activation='softmax')(x)

#     model = keras.Model(inputs=[input_watch, input_ear], outputs=output)

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model



# fedrep

def build_fedrep_model(num_classes=12):
    watch_input = keras.Input(shape=(20, 6), name="watch")
    x_watch = layers.Conv1D(64, 5, activation='relu', padding='same')(watch_input)
    x_watch = layers.MaxPooling1D(2)(x_watch)
    x_watch = layers.Flatten()(x_watch)

    ear_input = keras.Input(shape=(20, 6), name="ear")
    x_ear = layers.Conv1D(64, 5, activation='relu', padding='same')(ear_input)
    x_ear = layers.MaxPooling1D(2)(x_ear)
    x_ear = layers.Flatten()(x_ear)

    concatenated = layers.concatenate([x_watch, x_ear])
    shared_backbone = layers.Dense(128, activation='relu')(concatenated)
    head = layers.Dense(num_classes, activation='softmax', name="head")(shared_backbone)

    model = keras.Model(inputs=[watch_input, ear_input], outputs=head)
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
        X, y = get_dataset(filepath, window_size=20, stride=5, normalize=False)  # No normalization yet
        X_total.append(X)
        y_total.append(y)

    X = np.concatenate(X_total, axis=0)
    y = np.concatenate(y_total, axis=0)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split each input into watch and earable (first 6 and last 6 features)
    x_train_watch = x_train[:, :, :6]
    x_train_ear = x_train[:, :, 6:]

    x_test_watch = x_test[:, :, :6]
    x_test_ear = x_test[:, :, 6:]

    # Normalize separately
    x_train_watch, scaler_watch = normalize_features(x_train_watch, save=True, path="norm_watch.json")
    x_train_ear, scaler_ear = normalize_features(x_train_ear, save=True, path="norm_ear.json")

    x_test_watch = scaler_watch.transform(x_test_watch.reshape(-1, x_test_watch.shape[-1])).reshape(x_test_watch.shape)
    x_test_ear = scaler_ear.transform(x_test_ear.reshape(-1, x_test_ear.shape[-1])).reshape(x_test_ear.shape)

    # Return watch and earable inputs separately for model
    return [x_train_watch, x_train_ear], y_train, [x_test_watch, x_test_ear], y_test

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


def export_feature_stats(x_watch, x_ear, save_path="feature_stats.json"):
    """
    Save mean and std for both watch and earable sensor data.
    x_watch, x_ear: shape (samples, time, features=6)
    """
    watch_mean = x_watch.mean(axis=(0, 1)).tolist()
    watch_std = x_watch.std(axis=(0, 1)).tolist()

    ear_mean = x_ear.mean(axis=(0, 1)).tolist()
    ear_std = x_ear.std(axis=(0, 1)).tolist()

    stats = {
        "watch": {"mean": watch_mean, "std": watch_std},
        "earable": {"mean": ear_mean, "std": ear_std}
    }

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"[✓] Feature stats saved to: {save_path}")


# # def export_feature_stats(x_train, save_path="feature_stats.json"):
# #     # Compute mean and std per feature (across time and samples)
# #     feature_means = x_train.mean(axis=(0, 1)).tolist()
# #     feature_stds = x_train.std(axis=(0, 1)).tolist()
    
# #     stats = {"mean": feature_means, "std": feature_stds}
    
# #     with open(save_path, "w") as f:
# #         json.dump(stats, f, indent=4)
    
# #     print(f"[✓] Feature stats saved to: {save_path}")














# FEDPER


# import os
# import json
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sensorflow_model.dataset import get_dataset, normalize_features
# import torch
# from torch import nn


# # -------------------------
# # ✅ PyTorch ANN Model
# # -------------------------
# class ANN(nn.Module):
#     def __init__(self, args, name):
#         super(ANN, self).__init__()
#         self.name = name
#         self.len = 0
#         self.loss = 0

#         input_dim = 20 * 6  # each input: window of 20 timesteps × 6 features

#         # Watch branch
#         self.watch_fc1 = nn.Linear(input_dim, 128)
#         self.watch_fc2 = nn.Linear(128, 64)

#         # Earable branch
#         self.ear_fc1 = nn.Linear(input_dim, 128)
#         self.ear_fc2 = nn.Linear(128, 64)

#         # Head
#         self.head_fc1 = nn.Linear(64, 64)
#         self.head_fc2 = nn.Linear(64, 12)   # 12 activity classes

#         # Activations & dropout
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, data):
#         watch_input, ear_input = data

#         # Flatten inputs
#         watch_input = watch_input.view(watch_input.size(0), -1)
#         ear_input = ear_input.view(ear_input.size(0), -1)

#         # Watch branch
#         x_watch = self.watch_fc1(watch_input)
#         x_watch = self.sigmoid(x_watch)
#         x_watch = self.watch_fc2(x_watch)
#         x_watch = self.sigmoid(x_watch)
#         x_watch = self.dropout(x_watch)

#         # Earable branch
#         x_ear = self.ear_fc1(ear_input)
#         x_ear = self.sigmoid(x_ear)
#         x_ear = self.ear_fc2(x_ear)
#         x_ear = self.sigmoid(x_ear)
#         x_ear = self.dropout(x_ear)

#         # Combine by average
#         combined = (x_watch + x_ear) / 2

#         # Head
#         x = self.head_fc1(combined)
#         x = self.sigmoid(x)
#         x = self.dropout(x)
#         out = self.head_fc2(x)

#         return out


# # -------------------------
# # ✅ Utility functions
# # -------------------------
# def get_parameters(model):
#     # Convert model parameters to NumPy for FL
#     return [p.detach().cpu().numpy() for p in model.parameters()]

# def set_parameters(model, parameters):
#     # Load parameters back into model
#     for p, new_p in zip(model.parameters(), parameters):
#         p.data = torch.tensor(new_p, dtype=p.dtype)

# def load_data(partition_id, data_folder="myData"):
#     client_folder = os.path.join(data_folder, f"Client_{partition_id + 1}")
#     all_files = sorted([f for f in os.listdir(client_folder) if f.endswith(".csv")])

#     if not all_files:
#         raise ValueError(f"No CSV files found in folder {client_folder}")

#     X_total, y_total = [], []
#     for file in all_files:
#         filepath = os.path.join(client_folder, file)
#         X, y = get_dataset(filepath, window_size=20, stride=5, normalize=False)
#         X_total.append(X)
#         y_total.append(y)

#     X = np.concatenate(X_total, axis=0)
#     y = np.concatenate(y_total, axis=0)

#     # Train/test split
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Split into watch & earable features
#     x_train_watch = x_train[:, :, :6]
#     x_train_ear = x_train[:, :, 6:]
#     x_test_watch = x_test[:, :, :6]
#     x_test_ear = x_test[:, :, 6:]

#     # Normalize separately
#     x_train_watch, scaler_watch = normalize_features(x_train_watch, save=True, path="norm_watch.json")
#     x_train_ear, scaler_ear = normalize_features(x_train_ear, save=True, path="norm_ear.json")

#     x_test_watch = scaler_watch.transform(x_test_watch.reshape(-1, x_test_watch.shape[-1])).reshape(x_test_watch.shape)
#     x_test_ear = scaler_ear.transform(x_test_ear.reshape(-1, x_test_ear.shape[-1])).reshape(x_test_ear.shape)

#     return [x_train_watch, x_train_ear], y_train, [x_test_watch, x_test_ear], y_test

# def export_feature_stats(x_watch, x_ear, save_path="feature_stats.json"):
#     watch_mean = x_watch.mean(axis=(0, 1)).tolist()
#     watch_std = x_watch.std(axis=(0, 1)).tolist()
#     ear_mean = x_ear.mean(axis=(0, 1)).tolist()
#     ear_std = x_ear.std(axis=(0, 1)).tolist()

#     stats = {
#         "watch": {"mean": watch_mean, "std": watch_std},
#         "earable": {"mean": ear_mean, "std": ear_std}
#     }

#     with open(save_path, "w") as f:
#         json.dump(stats, f, indent=4)
#     print(f"[✓] Feature stats saved to: {save_path}")



