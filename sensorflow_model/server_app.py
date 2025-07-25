# from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from typing import Dict
# import matplotlib.pyplot as plt
# from flwr.server.strategy import FedAvg
# from sensorflow_model.personalize import evaluate_personalization_on_clients
# # evaluate_personalization_on_clients(self.model, k=5)
# from sensorflow_model.task import (
#     load_model,
#     get_parameters,
#     set_parameters,
#     export_to_tflite, 
# )

# import json
# import os
# import numpy as np
# os.environ["RAY_DISABLE_DASHBOARD"] = "1"



# ACTIVITY_CLASSES = {
#     0: "Sitting + Typing on Desk",
#     1: "Sitting + Taking Notes", 
#     2: "Standing + Writing on Whiteboard",
#     3: "Standing + Erasing Whiteboard",
#     4: "Sitting + Talking + Waving Hands",
#     5: "Standing + Talking + Waving Hands",
#     6: "Sitting + Drinking Water",
#     7: "Sitting + Drinking Coffee",
#     8: "Standing + Drinking Water",
#     9: "Standing + Drinking Coffee",
#     10: "Sitting + Scrolling on Phone",
# }

# def aggregate_fit_metrics(metrics) -> Dict[str, Scalar]:
#     total_examples = sum(num_examples for num_examples, _ in metrics)
#     avg_accuracy = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
#     avg_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
#     return {"accuracy": avg_accuracy, "loss": avg_loss}


# def weighted_average(metrics) -> Dict[str, Scalar]:
#     total_examples = sum(num_examples for num_examples, _ in metrics)
#     weighted_result = sum(
#         num_examples * metric["accuracy"]
#         for num_examples, metric in metrics
#     ) / total_examples
#     return {"accuracy ggggggggggggggg": float(weighted_result)}



# # class SaveFedAvg(FedAvg):
# #     def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
# #         super().__init__(**kwargs)
# #         self.model = model
# #         self.export_path = export_path
# #         self.tflite_path = tflite_path
# #         self.max_rounds = max_rounds

# #     # def aggregate_fit(self, server_round, results, failures):
# #     #     aggregated_result = super().aggregate_fit(server_round, results, failures)

# #     #     if aggregated_result is not None:
# #     #         aggregated_parameters, aggregated_metrics = aggregated_result

# #     #         #  Set weights and save at final round
# #     #         if server_round == self.max_rounds and aggregated_parameters is not None:
# #     #             weights = parameters_to_ndarrays(aggregated_parameters)
# #     #             self.model.set_weights(weights)

# #     #             #  Save in Keras format
# #     #             self.model.save(self.export_path)
# #     #             print(f" Saved Keras model to {self.export_path}")

# #     #             if self.tflite_path:
# #     #                 export_to_tflite(self.model, self.tflite_path)
# #     #                 print(f" Exported model to TFLite: {self.tflite_path}")
                
# #     #             #evaluate_personalization_on_clients(self.model, k=10)
# #     #             evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

# #     #         return aggregated_parameters, aggregated_metrics

# #         return None

# class SaveFedAvg(FedAvg):
#     def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.export_path = export_path
#         self.tflite_path = tflite_path
#         self.max_rounds = max_rounds
#         self.masks = []  # Store simulated client masks
#         self.loss_history = []
#         self.acc_history = []


#     # def aggregate_fit(self, server_round, results, failures):
#     #     if not results:
#     #         return None, {}

#     #     try:
#     #         # Unpack weights
#     #         weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

#     #         # Simulate random masks
#     #         simulated_masks = [self._simulate_client_mask(w) for w in weights_ndarrays]

#     #         # Average masked weights
#     #         masked_avg = [np.mean(layer_group, axis=0) for layer_group in zip(*weights_ndarrays)]

#     #         # Average masks
#     #         avg_mask = [np.mean(layer_group, axis=0) for layer_group in zip(*simulated_masks)]

#     #         # Subtract mask from masked average to get unmasked weights
#     #         unmasked_avg = [masked - mask for masked, mask in zip(masked_avg, avg_mask)]
#     #         aggregated_parameters = ndarrays_to_parameters(unmasked_avg)

#     #         # Save final model if this is the last round
#     #         if server_round == self.max_rounds and aggregated_parameters is not None:
#     #             self.model.set_weights(unmasked_avg)
#     #             self.model.save(self.export_path)
#     #             print(f"✅ Saved Keras model to {self.export_path}")

#     #             if self.tflite_path:
#     #                 export_to_tflite(self.model, self.tflite_path)
#     #                 print(f"✅ Exported model to TFLite: {self.tflite_path}")

#     #             evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

#     #     except Exception as e:
#     #         print(f"❌ Error during aggregation: {e}")
#     #         return None, {}

#     #     # Safely aggregate metrics
#     #     try:
#     #         aggregated_metrics = self.aggregate_fit_metrics([
#     #             (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
#     #         ])
#     #     except AttributeError:
#     #         print("⚠️ Warning: `aggregate_fit_metrics` not defined. Returning empty metrics.")
#     #         aggregated_metrics = {}

#     #     # Save history
#     #     # Save history
#     #     if "loss" in aggregated_metrics:
#     #         self.loss_history.append((server_round, aggregated_metrics["loss"]))
#     #     if "accuracy" in aggregated_metrics:
#     #         self.acc_history.append((server_round, aggregated_metrics["accuracy"]))

#     #     # Final round: Save convergence plot
#     #     if server_round == self.max_rounds:
#     #         try:
#     #             rounds_loss = [r for r, _ in self.loss_history]
#     #             loss_values = [l for _, l in self.loss_history]

#     #             rounds_acc = [r for r, _ in self.acc_history]
#     #             acc_values = [a for _, a in self.acc_history]

#     #             plt.figure(figsize=(10, 4))

#     #             plt.subplot(1, 2, 1)
#     #             plt.plot(rounds_loss, loss_values, marker='o', label='Loss')
#     #             plt.title("Loss over Rounds")
#     #             plt.xlabel("Round")
#     #             plt.ylabel("Loss")
#     #             plt.grid(True)

#     #             plt.subplot(1, 2, 2)
#     #             plt.plot(rounds_acc, acc_values, marker='o', color='green', label='Accuracy')
#     #             plt.title("Accuracy over Rounds")
#     #             plt.xlabel("Round")
#     #             plt.ylabel("Accuracy")
#     #             plt.grid(True)

#     #             plt.tight_layout()
#     #             plt.savefig("convergence_plot.png")
#     #             print("📈 Saved convergence plot to convergence_plot.png")
#     #         except Exception as e:
#     #             print(f"⚠️ Could not generate convergence plot: {e}")


#     #     return aggregated_parameters, aggregated_metrics
#     def aggregate_fit(self, server_round, results, failures):
#         if not results:
#             return None, {}

#         try:
#             # Unpack weights
#             weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

#             # Simulate random masks
#             simulated_masks = [self._simulate_client_mask(w) for w in weights_ndarrays]

#             # Average masked weights
#             masked_avg = [np.mean(layer_group, axis=0) for layer_group in zip(*weights_ndarrays)]

#             # Average masks
#             avg_mask = [np.mean(layer_group, axis=0) for layer_group in zip(*simulated_masks)]

#             # Subtract mask from masked average to get unmasked weights
#             unmasked_avg = [masked - mask for masked, mask in zip(masked_avg, avg_mask)]
#             aggregated_parameters = ndarrays_to_parameters(unmasked_avg)

#             # Set final weights and export model if this is the last round
#             if server_round == self.max_rounds and aggregated_parameters is not None:
#                 self.model.set_weights(unmasked_avg)
#                 self.model.save(self.export_path)
#                 print(f"✅ Saved Keras model to {self.export_path}")

#                 if self.tflite_path:
#                     export_to_tflite(self.model, self.tflite_path)
#                     print(f"✅ Exported model to TFLite: {self.tflite_path}")

#                 # Trigger few-shot personalization evaluation
#                 evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

#         except Exception as e:
#             print(f"❌ Error during aggregation: {e}")
#             return None, {}

#         # Aggregate client metrics
#         try:
#             aggregated_metrics = self.aggregate_fit_metrics([
#                 (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
#             ])
#         except AttributeError:
#             print("⚠️ Warning: `aggregate_fit_metrics` not defined. Returning empty metrics.")
#             aggregated_metrics = {}

#         # === Lazy Init for convergence history tracking ===
#         if not hasattr(self, "loss_history"):
#             self.loss_history = []
#         if not hasattr(self, "acc_history"):
#             self.acc_history = []

#         # # Save convergence metrics
#         # if "loss" in aggregated_metrics:
#         #     self.loss_history.append((server_round, aggregated_metrics["loss"]))
#         # if "accuracy" in aggregated_metrics:
#         #     self.acc_history.append((server_round, aggregated_metrics["accuracy"]))
#         if "loss" in aggregated_metrics:
#             self.loss_history.append((server_round, aggregated_metrics["loss"]))
#         if "accuracy" in aggregated_metrics:
#             self.acc_history.append((server_round, aggregated_metrics["accuracy"]))


#         return aggregated_parameters, aggregated_metrics


#     def aggregate_fit_metrics(self, results):
#     #"""Aggregate client metrics using weighted average."""
#         if not results:
#             return {}

#         metrics = {}
#         total_examples = sum(num_examples for num_examples, _ in results)

#         for num_examples, client_metrics in results:
#             for key, value in client_metrics.items():
#                 if key not in metrics:
#                     metrics[key] = 0.0
#                 metrics[key] += value * (num_examples / total_examples)

#         return metrics


#     def _simulate_client_mask(self, weights):
#         """Simulate a random mask of the same shape as model weights."""
#         return [np.random.normal(0, 0.01, w.shape).astype(np.float32) for w in weights]
#     import matplotlib.pyplot as plt

#     def plot_convergence(self, save_path="convergence_plot.png"):
#         if not hasattr(self, "loss_history") or not hasattr(self, "acc_history"):
#             print("⚠️ No convergence history found. Cannot plot.")
#             return

#         rounds_loss = [r for r, _ in self.loss_history]
#         loss_values = [l for _, l in self.loss_history]

#         rounds_acc = [r for r, _ in self.acc_history]
#         acc_values = [a for _, a in self.acc_history]

#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 4))

#         plt.subplot(1, 2, 1)
#         plt.plot(rounds_loss, loss_values, marker='o', label='Loss')
#         plt.title("Loss over Rounds")
#         plt.xlabel("Round")
#         plt.ylabel("Loss")
#         plt.grid(True)

#         plt.subplot(1, 2, 2)
#         plt.plot(rounds_acc, acc_values, marker='o', color='green', label='Accuracy')
#         plt.title("Accuracy over Rounds")
#         plt.xlabel("Round")
#         plt.ylabel("Accuracy")
#         plt.grid(True)

#         plt.tight_layout()
#         plt.savefig(save_path)
#         print(f"📈 Saved convergence plot to {save_path}")
#         plt.close()

#         # ✅ Save raw convergence data to JSON
#         with open("convergence_history.json", "w") as f:
#             json.dump({
#                 "loss": self.loss_history,
#                 "accuracy": self.acc_history,
#             }, f)
#             print("📝 Saved convergence history to convergence_history.json")




# def save_normalization_stats(mean, std, path="normalization.json"):
#     with open(path, "w") as f:
#         json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
#     print(f"Saved normalization stats to {path}")





# def server_fn(context: Context) -> ServerAppComponents:
#     num_rounds = int(context.run_config["num-server-rounds"])

#     model = load_model()
#     weights = get_parameters(model)
#     parameters = ndarrays_to_parameters(weights)

#     strategy = SaveFedAvg(
#         model=model,
#         export_path="final_model.keras",
#         tflite_path="final_model.tflite",
#         max_rounds=num_rounds,
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_available_clients=2,
#         initial_parameters=parameters,
#         fit_metrics_aggregation_fn=aggregate_fit_metrics,
#         evaluate_metrics_aggregation_fn=weighted_average,
#     )

#     config = ServerConfig(num_rounds=num_rounds)
#     return ServerAppComponents(strategy=strategy, config=config)


# app = ServerApp(server_fn=server_fn)

# # if __name__ == "__main__":
# #     app = ServerApp(server_fn=server_fn)
# #     history = app.run()
    
# #     # Plot convergence
# #     try:
# #         losses = history.losses_distributed  # [(round, loss), ...]
# #         accuracies = history.metrics_distributed.get("accuracy ggggggggggggggg", [])

# #         rounds_loss = [r for r, _ in losses]
# #         loss_values = [l for _, l in losses]

# #         rounds_acc = [r for r, _ in accuracies]
# #         acc_values = [a for _, a in accuracies]

# #         plt.figure(figsize=(10, 4))

# #         plt.subplot(1, 2, 1)
# #         plt.plot(rounds_loss, loss_values, marker='o', label='Loss')
# #         plt.title("Loss over Rounds")
# #         plt.xlabel("Round")
# #         plt.ylabel("Loss")
# #         plt.grid(True)

# #         plt.subplot(1, 2, 2)
# #         plt.plot(rounds_acc, acc_values, marker='o', color='green', label='Accuracy')
# #         plt.title("Accuracy over Rounds")
# #         plt.xlabel("Round")
# #         plt.ylabel("Accuracy")
# #         plt.grid(True)

# #         plt.tight_layout()
# #         plt.savefig("convergence_plot.png")
# #         print("📈 Saved convergence plot to convergence_plot.png")

# #         # Optional: show the plot live if running interactively
# #         plt.show()

# #     except Exception as e:
# #         print(f"⚠️ Could not generate convergence plot: {e}")



import os
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

import matplotlib
matplotlib.use("Agg")  # 🛠️ Fix for GUI error in headless/threaded mode
import matplotlib.pyplot as plt
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import Dict

from flwr.server.strategy import FedAvg, FedAvgM
from sensorflow_model.personalize import evaluate_personalization_on_clients

from sensorflow_model.task import (
    load_model,
    get_parameters,
    set_parameters,
    export_to_tflite, 
)

import json
import os
import numpy as np

os.environ["RAY_DISABLE_DASHBOARD"] = "1"

ACTIVITY_CLASSES = {
    0: "Sitting + Typing on Desk",
    1: "Sitting + Taking Notes", 
    2: "Standing + Writing on Whiteboard",
    3: "Standing + Erasing Whiteboard",
    4: "Sitting + Talking + Waving Hands",
    5: "Standing + Talking + Waving Hands",
    6: "Sitting + Drinking Water",
    7: "Sitting + Drinking Coffee",
    8: "Standing + Drinking Water",
    9: "Standing + Drinking Coffee",
    10: "Sitting + Scrolling on Phone",
}

def aggregate_fit_metrics(metrics) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    avg_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
    return {"accuracy": avg_accuracy, "loss": avg_loss}

def weighted_average(metrics) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_result = sum(
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ) / total_examples
    return {"accuracy ggggggggggggggg": float(weighted_result)}

class SaveFedAvg(FedAvg):
    def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.export_path = export_path
        self.tflite_path = tflite_path
        self.max_rounds = max_rounds
        self.loss_history = []
        self.acc_history = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        try:
            weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            simulated_masks = [self._simulate_client_mask(w) for w in weights_ndarrays]

            masked_avg = [np.mean(layer_group, axis=0) for layer_group in zip(*weights_ndarrays)]
            avg_mask = [np.mean(layer_group, axis=0) for layer_group in zip(*simulated_masks)]

            unmasked_avg = [masked - mask for masked, mask in zip(masked_avg, avg_mask)]
            aggregated_parameters = ndarrays_to_parameters(unmasked_avg)

            if server_round == self.max_rounds and aggregated_parameters is not None:
                self.model.set_weights(unmasked_avg)
                self.model.save(self.export_path)
                print(f"✅ Saved Keras model to {self.export_path}")

                if self.tflite_path:
                    export_to_tflite(self.model, self.tflite_path)
                    print(f"✅ Exported model to TFLite: {self.tflite_path}")

                evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
            return None, {}

        try:
            aggregated_metrics = self.aggregate_fit_metrics([
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ])
        except AttributeError:
            print("⚠️ Warning: `aggregate_fit_metrics` not defined. Returning empty metrics.")
            aggregated_metrics = {}

        if "loss" in aggregated_metrics:
            self.loss_history.append((server_round, aggregated_metrics["loss"]))
        if "accuracy" in aggregated_metrics:
            self.acc_history.append((server_round, aggregated_metrics["accuracy"]))

        # Final round: Save convergence plot and history
        if server_round == self.max_rounds:
            self.plot_convergence()

        return aggregated_parameters, aggregated_metrics


    def aggregate_fit_metrics(self, results):
        if not results:
            return {}
        metrics = {}
        total_examples = sum(num_examples for num_examples, _ in results)
        for num_examples, client_metrics in results:
            for key, value in client_metrics.items():
                if key not in metrics:
                    metrics[key] = 0.0
                metrics[key] += value * (num_examples / total_examples)
        return metrics

    def _simulate_client_mask(self, weights):
        return [np.random.normal(0, 0.01, w.shape).astype(np.float32) for w in weights]

    def plot_convergence(self, save_path="convergence_plot_50.png"):
        if not hasattr(self, "loss_history") or not hasattr(self, "acc_history"):
            print("⚠️ No convergence history found. Cannot plot.")
            return

        rounds_loss = [r for r, _ in self.loss_history]
        loss_values = [l for _, l in self.loss_history]

        rounds_acc = [r for r, _ in self.acc_history]
        acc_values = [a for _, a in self.acc_history]

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(rounds_loss, loss_values, marker='o', label='Loss')
        plt.title("Loss over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(rounds_acc, acc_values, marker='o', color='green', label='Accuracy')
        plt.title("Accuracy over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Saved convergence plot to {save_path}")
        plt.close()

        with open("convergence_history_50.json", "w") as f:
            json.dump({
                "loss": self.loss_history,
                "accuracy": self.acc_history,
            }, f)
            print("📝 Saved convergence history to convergence_history.json")

def save_normalization_stats(mean, std, path="normalization.json"):
    with open(path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"Saved normalization stats to {path}")

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config["num-server-rounds"])
    model = load_model()
    weights = get_parameters(model)
    parameters = ndarrays_to_parameters(weights)

    strategy = SaveFedAvg(
        model=model,
        export_path="final_model.keras",
        tflite_path="final_model.tflite",
        max_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
