import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode


def load_csv(filepath):
    """
    Load sensor data CSV and return DataFrame.
    """
    df = pd.read_csv(filepath)
    
    df.fillna(0, inplace=True)
    
    return df


def normalize_features(X):
    """
    Normalize features using z-score normalization.
    """
    original_shape = X.shape
    
    X_flat = X.reshape(-1, X.shape[-1])
    
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X_flat)
    
    return X_scaled.reshape(original_shape), scaler


def create_sliding_windows(df, window_size=200, stride=100):
    """
    Create sliding windows from raw sensor DataFrame.
    Returns (X, y) arrays.
    """
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    data = df[feature_cols].values
    labels = df[label_col].values

    X_windows = []
    y_windows = []

    for start in range(0, len(data) - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]
        label_window = labels[start:end]

        if len(window) == window_size:
            X_windows.append(window)
            y_windows.append(mode(label_window, keepdims=False).mode)

    X = np.array(X_windows)
    
    y = np.array(y_windows).astype(np.int32)
    
    return X, y


def get_dataset(filepath, window_size=200, stride=100, normalize=True):
    """
    Full dataset pipeline: Load CSV → Window → Normalize → Return X, y (and optionally scaler)
    """
    df = load_csv(filepath)
    X, y = create_sliding_windows(df, window_size, stride)

    if normalize:
        X, scaler = normalize_features(X)
    
    return X, y
