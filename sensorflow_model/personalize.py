# sensorflow_model/personalize.py
import os 
import numpy as np
from tensorflow import keras
from sensorflow_model.task import load_data


def get_few_shot_data(x_train, y_train, k=10, num_classes=12):
    """
    Sample k examples per class from x_train/y_train for few-shot personalization.
    """
    x_few, y_few = [], []
    for class_id in range(num_classes):
        idx = np.where(y_train == class_id)[0]
        if len(idx) == 0:
            continue  # skip if class is missing for the user
        selected = np.random.choice(idx, size=min(k, len(idx)), replace=False)
        x_few.append(x_train[selected])
        y_few.append(y_train[selected])
    return np.concatenate(x_few), np.concatenate(y_few)


def personalize_model(global_model, x_few, y_few, x_test, y_test,
                      fine_tune_all=False, epochs=5, batch_size=16):
    """
    Clone and fine-tune the global model on few-shot data, then evaluate.
    """
    model = keras.models.clone_model(global_model)
    model.set_weights(global_model.get_weights())

    if not fine_tune_all:
        for layer in model.layers[:-2]:  # Freeze encoder layers
            layer.trainable = False

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_few, y_few, epochs=epochs, batch_size=batch_size, verbose=0)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc


# def evaluate_personalization_on_clients(global_model, data_folder="myData", k=10):
#     """
#     Evaluate personalization for each client using K-shot adaptation.
#     """
#     print(f"\n\U0001F4CA Evaluating Few-Shot Personalization (K={k})\n")

#     for client_id in range(6):  # assuming 8 clients
#         x_train, y_train, x_test, y_test = load_data(client_id, data_folder)
#         acc_before = global_model.evaluate(x_test, y_test, verbose=0)[1]

#         x_few, y_few = get_few_shot_data(x_train, y_train, k=k)
#         acc_after = personalize_model(global_model, x_few, y_few, x_test, y_test, fine_tune_all=False)

#         print(f"[Client {client_id + 1}] Accuracy Before: {acc_before:.4f} | After Personalization: {acc_after:.4f}")


# def evaluate_personalization_on_clients(global_model, data_folder="myData", k_values=[5, 10, 20]):
#     """
#     Evaluate personalization for each client using multiple K-shot settings.
#     """
#     print(f"\n📊 Evaluating Few-Shot Personalization for K in {k_values}\n")

#     for k in k_values:
#         print(f"\n🧪 K = {k} Examples Per Class\n")
#         for client_id in range(6):  # assuming 6 clients
#             x_train, y_train, x_test, y_test = load_data(client_id, data_folder)
#             acc_before = global_model.evaluate(x_test, y_test, verbose=0)[1]

#             x_few, y_few = get_few_shot_data(x_train, y_train, k=k)
#             if len(x_few) == 0:
#                 print(f"[Client {client_id + 1}] Skipped: No data available for K={k}")
#                 continue

#             acc_after = personalize_model(global_model, x_few, y_few, x_test, y_test, fine_tune_all=False)

#             print(f"[Client {client_id + 1}] Accuracy Before: {acc_before:.4f} | After Personalization: {acc_after:.4f}")

# def evaluate_personalization_on_clients(global_model, data_folder="myData", k_values=[5, 10, 20]):
#     from sensorflow_model.task import load_data, load_model, set_parameters
#     from sklearn.metrics import accuracy_score
#     import numpy as np

#     client_dirs = sorted([f for f in os.listdir(data_folder) if f.startswith("Client_")])
    
#     for k in k_values:
#         print(f"\n🧪 K = {k} Examples Per Class")

#         accuracies = []

#         for i, _ in enumerate(client_dirs):
#             model = load_model()
#             set_parameters(model, global_model.get_weights())

#             [x_train_watch, x_train_ear], y_train, [x_test_watch, x_test_ear], y_test = load_data(i)

#             # Collect k examples per class
#             x_few_watch, x_few_ear, y_few = [], [], []

#             classes = np.unique(y_train)
#             for cls in classes:
#                 idx = np.where(y_train == cls)[0]
#                 if len(idx) >= k:
#                     selected = idx[:k]
#                     x_few_watch.append(x_train_watch[selected])
#                     x_few_ear.append(x_train_ear[selected])
#                     y_few.append(y_train[selected])

#             if len(x_few_watch) == 0:
#                 continue

#             x_few_watch = np.concatenate(x_few_watch)
#             x_few_ear = np.concatenate(x_few_ear)
#             y_few = np.concatenate(y_few)

#             # Fine-tune
#             model.fit([x_few_watch, x_few_ear], y_few, epochs=3, batch_size=16, verbose=0)

#             # Evaluate
#             y_pred = np.argmax(model.predict([x_test_watch, x_test_ear], verbose=0), axis=1)
#             acc = accuracy_score(y_test, y_pred)
#             accuracies.append(acc)

#         if accuracies:
#             mean_acc = np.mean(accuracies)
#             print(f"📈 Avg personalization accuracy (K={k}): {mean_acc:.4f}")
#         else:
#             print("⚠️ No clients had enough examples per class.")
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sensorflow_model.task import load_data, load_model, set_parameters

def evaluate_personalization_on_clients(global_model, data_folder="myData", k_values=[5, 10, 20]):
    client_dirs = sorted([f for f in os.listdir(data_folder) if f.startswith("Client_")])
    
    for k in k_values:
        print(f"\n🧪 K = {k} Examples Per Class")
        accuracies = []

        for client_id, _ in enumerate(client_dirs):
            model = load_model()
            set_parameters(model, global_model.get_weights())

            [x_train_watch, x_train_ear], y_train, [x_test_watch, x_test_ear], y_test = load_data(client_id)

            # Collect k examples per class
            x_few_watch, x_few_ear, y_few = [], [], []
            classes = np.unique(y_train)
            for cls in classes:
                idx = np.where(y_train == cls)[0]
                if len(idx) >= k:
                    selected = idx[:k]
                    x_few_watch.append(x_train_watch[selected])
                    x_few_ear.append(x_train_ear[selected])
                    y_few.append(y_train[selected])

            if len(x_few_watch) == 0:
                continue

            x_few_watch = np.concatenate(x_few_watch)
            x_few_ear = np.concatenate(x_few_ear)
            y_few = np.concatenate(y_few)

            # Accuracy before personalization
            y_pred_before = np.argmax(model.predict([x_test_watch, x_test_ear], verbose=0), axis=1)
            acc_before = accuracy_score(y_test, y_pred_before)

            # Personalization (fine-tune on few-shot)
            model.fit([x_few_watch, x_few_ear], y_few, epochs=3, batch_size=16, verbose=0)

            # Accuracy after personalization
            y_pred_after = np.argmax(model.predict([x_test_watch, x_test_ear], verbose=0), axis=1)
            acc_after = accuracy_score(y_test, y_pred_after)

            accuracies.append(acc_after)

            print(f"[Client {client_id + 1}] Accuracy Before: {acc_before:.4f} | After Personalization: {acc_after:.4f}")

        if accuracies:
            mean_acc = np.mean(accuracies)
            print(f"\n📈 Avg personalization accuracy (K={k}): {mean_acc:.4f}")
        else:
            print("⚠️ No clients had enough examples per class.")
