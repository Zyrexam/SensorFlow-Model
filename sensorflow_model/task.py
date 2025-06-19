import os

import keras
from keras import layers
from dataset import get_dataset
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



def load_model():
    model = keras.Sequential([
        keras.Input(shape=(200, 12)),  # 2s window with 12 features
        
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.LSTM(64),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(12, activation='softmax')  # 12 activity classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_parameters(model):
    return model.get_weights()


def set_parameters(model, parameters):
    model.set_weights(parameters)



def load_data(partition_id, data_folder="Data"):
    all_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".csv")])
    
    if len(all_files) == 0:
        raise ValueError("No CSV files found in the data folder!")

    # Safely get file by partition index
    if partition_id >= len(all_files):
        raise IndexError(f"Only {len(all_files)} CSV files found. Partition ID {partition_id} is out of range.")

    filepath = os.path.join(data_folder, all_files[partition_id])

    
    X, y = get_dataset(filepath, window_size=200, stride=100)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test




def export_to_tflite(model, output_path="model.tflite"):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model) # type: ignore





