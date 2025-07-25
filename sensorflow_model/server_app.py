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






# Best working with masking and personlization code 

# import os
# os.environ["RAY_DISABLE_DASHBOARD"] = "1"

# import matplotlib
# matplotlib.use("Agg")  
# import matplotlib.pyplot as plt
# from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from typing import Dict

# from flwr.server.strategy import FedAvg, FedAvgM
# from sensorflow_model.personalize import evaluate_personalization_on_clients
# # import torch
# from sensorflow_model.task import (
#     load_model,
#     get_parameters,
#     set_parameters,
#     export_to_tflite, 
# )
# # from sensorflow_model.task import ANN

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

# class SaveFedAvg(FedAvg):
#     def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.export_path = export_path
#         self.tflite_path = tflite_path
#         self.max_rounds = max_rounds
#         self.loss_history = []
#         self.acc_history = []

#     def aggregate_fit(self, server_round, results, failures):
#         if not results:
#             return None, {}

#         try:
#             weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
#             simulated_masks = [self._simulate_client_mask(w) for w in weights_ndarrays]

#             masked_avg = [np.mean(layer_group, axis=0) for layer_group in zip(*weights_ndarrays)]
#             avg_mask = [np.mean(layer_group, axis=0) for layer_group in zip(*simulated_masks)]

#             unmasked_avg = [masked - mask for masked, mask in zip(masked_avg, avg_mask)]
#             aggregated_parameters = ndarrays_to_parameters(unmasked_avg)

#             if server_round == self.max_rounds and aggregated_parameters is not None:
#                 self.model.set_weights(unmasked_avg)
#                 self.model.save(self.export_path)
#                 print(f"✅ Saved Keras model to {self.export_path}")

#                 if self.tflite_path:
#                     export_to_tflite(self.model, self.tflite_path)
#                     print(f"✅ Exported model to TFLite: {self.tflite_path}")

#                 evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

#         except Exception as e:
#             print(f"❌ Error during aggregation: {e}")
#             return None, {}

#         try:
#             aggregated_metrics = self.aggregate_fit_metrics([
#                 (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
#             ])
#         except AttributeError:
#             print("⚠️ Warning: `aggregate_fit_metrics` not defined. Returning empty metrics.")
#             aggregated_metrics = {}

#         if "loss" in aggregated_metrics:
#             self.loss_history.append((server_round, aggregated_metrics["loss"]))
#         if "accuracy" in aggregated_metrics:
#             self.acc_history.append((server_round, aggregated_metrics["accuracy"]))

#         # Final round: Save convergence plot and history
#         if server_round == self.max_rounds:
#             self.plot_convergence()

#         return aggregated_parameters, aggregated_metrics


#     def aggregate_fit_metrics(self, results):
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
#         return [np.random.normal(0, 0.01, w.shape).astype(np.float32) for w in weights]

#     def plot_convergence(self, save_path="convergence_plot_50.png"):
#         if not hasattr(self, "loss_history") or not hasattr(self, "acc_history"):
#             print("⚠️ No convergence history found. Cannot plot.")
#             return

#         rounds_loss = [r for r, _ in self.loss_history]
#         loss_values = [l for _, l in self.loss_history]

#         rounds_acc = [r for r, _ in self.acc_history]
#         acc_values = [a for _, a in self.acc_history]

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

#         with open("convergence_history_50.json", "w") as f:
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









# FEDPROX




# import os
# os.environ["RAY_DISABLE_DASHBOARD"] = "1"

# import matplotlib
# matplotlib.use("Agg")  
# import matplotlib.pyplot as plt
# from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from typing import Dict

# from flwr.server.strategy import FedAvg, FedAvgM
# from sensorflow_model.personalize import evaluate_personalization_on_clients
# # import torch
# from sensorflow_model.task import (
#     load_model,
#     get_parameters,
#     set_parameters,
#     export_to_tflite, 
# )
# # from sensorflow_model.task import ANN

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

# class SaveFedProx(FedAvg):
#     def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, proximal_mu=0.01, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.export_path = export_path
#         self.tflite_path = tflite_path
#         self.max_rounds = max_rounds
#         self.proximal_mu = proximal_mu
#         self.loss_history = []
#         self.acc_history = []

#     def aggregate_fit(self, server_round, results, failures):
#         if not results:
#             return None, {}

#         try:
#             # Convert client parameters to ndarray
#             weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

#             # Compute mean parameters (FedAvg aggregation)
#             avg_parameters = [np.mean(layer, axis=0) for layer in zip(*weights_ndarrays)]

#             # Apply FedProx proximal term (this is conceptual; in practice the clients apply the proximal regularization)
#             # On server side, we typically still aggregate normally, but we can log that we're using FedProx
#             print(f"🔧 FedProx: proximal_mu={self.proximal_mu}")

#             # Update model and export
#             aggregated_parameters = ndarrays_to_parameters(avg_parameters)
#             if server_round == self.max_rounds:
#                 self.model.set_weights(avg_parameters)
#                 self.model.save(self.export_path)
#                 print(f"✅ Saved Keras model to {self.export_path}")
#                 if self.tflite_path:
#                     export_to_tflite(self.model, self.tflite_path)
#                     print(f"✅ Exported model to TFLite: {self.tflite_path}")
#                 # evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

#         except Exception as e:
#             print(f"❌ Error during aggregation: {e}")
#             return None, {}

#         # Aggregate metrics
#         try:
#             aggregated_metrics = self.aggregate_fit_metrics([
#                 (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
#             ])
#         except AttributeError:
#             aggregated_metrics = {}

#         # Save for convergence plot
#         if "loss" in aggregated_metrics:
#             self.loss_history.append((server_round, aggregated_metrics["loss"]))
#         if "accuracy" in aggregated_metrics:
#             self.acc_history.append((server_round, aggregated_metrics["accuracy"]))

#         # Final round: save convergence plot
#         if server_round == self.max_rounds:
#             self.plot_convergence()

#         return aggregated_parameters, aggregated_metrics
    
    
#     def aggregate_fit_metrics(self, results):
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
#         return [np.random.normal(0, 0.01, w.shape).astype(np.float32) for w in weights]

#     def plot_convergence(self, save_path="fedprox_convergence_plot_10.png"):
#         if not hasattr(self, "loss_history") or not hasattr(self, "acc_history"):
#             print("⚠️ No convergence history found. Cannot plot.")
#             return

#         rounds_loss = [r for r, _ in self.loss_history]
#         loss_values = [l for _, l in self.loss_history]

#         rounds_acc = [r for r, _ in self.acc_history]
#         acc_values = [a for _, a in self.acc_history]

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

#         with open("fedprox_convergence_history_10.json", "w") as f:
#             json.dump({
#                 "loss": self.loss_history,
#                 "accuracy": self.acc_history,
#             }, f)
#             print("📝 Saved convergence history to fedprox_convergence_history_10.json")

# def save_normalization_stats(mean, std, path="normalization.json"):
#     with open(path, "w") as f:
#         json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
#     print(f"Saved normalization stats to {path}")

# def server_fn(context: Context) -> ServerAppComponents:
#     num_rounds = int(context.run_config["num-server-rounds"])
#     model = load_model()
#     weights = get_parameters(model)
#     parameters = ndarrays_to_parameters(weights)

#     strategy = SaveFedProx(
#         model=model,
#         export_path="final_model.keras",
#         tflite_path="final_model.tflite",
#         max_rounds=num_rounds,
#         proximal_mu=0.01,    # your FedProx parameter
#         fraction_fit=1.0,
#         fraction_evaluate=0.5,
#         min_available_clients=2,
#         initial_parameters=parameters,
#         fit_metrics_aggregation_fn=aggregate_fit_metrics,
#         evaluate_metrics_aggregation_fn=weighted_average,
#     )

#     config = ServerConfig(num_rounds=num_rounds)
#     return ServerAppComponents(strategy=strategy, config=config)


# app = ServerApp(server_fn=server_fn)








# FEDREP 

import os
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np

from typing import Dict
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from sensorflow_model.task import build_fedrep_model, get_parameters, export_to_tflite

def aggregate_fit_metrics(metrics) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    avg_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
    return {"accuracy": avg_accuracy, "loss": avg_loss}

def weighted_average(metrics) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_result = sum(num_examples * metric["accuracy"] for num_examples, metric in metrics) / total_examples
    return {"accuracy": float(weighted_result)}


class SaveFedRep(FedAvg):
    def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, proximal_mu=0.01, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.export_path = export_path
        self.tflite_path = tflite_path
        self.max_rounds = max_rounds
        self.proximal_mu = proximal_mu

        # Histories
        self.train_loss_history = []
        self.train_acc_history = []
        self.eval_loss_history = []
        self.eval_acc_history = []

    def aggregate_fit_metrics(self, results):
        """Aggregate metrics weighted by number of examples."""
        if not results:
            return {}
        total_examples = sum(num_examples for num_examples, _ in results)
        metrics = {}
        for num_examples, client_metrics in results:
            for k, v in client_metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v * (num_examples / total_examples)
        return metrics

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Aggregate backbone weights
        weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        avg_parameters = [np.mean(layer, axis=0) for layer in zip(*weights_ndarrays)]
        aggregated_parameters = ndarrays_to_parameters(avg_parameters)
        self.model.set_weights(avg_parameters)

        # Aggregate training metrics returned by clients
        aggregated_metrics = self.aggregate_fit_metrics([
            (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
        ])

        # Save to history
        if "loss" in aggregated_metrics:
            self.train_loss_history.append((server_round, aggregated_metrics["loss"]))
        if "accuracy" in aggregated_metrics:
            self.train_acc_history.append((server_round, aggregated_metrics["accuracy"]))

        print(f"🛠️ Round {server_round} train metrics: {aggregated_metrics}")

        # On last round, save training JSON & plot
        if server_round == self.max_rounds:
            self.save_train_history_json()
            self.plot_train_convergence()

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        # Aggregate eval metrics returned by clients
        aggregated_metrics = self.aggregate_fit_metrics([
            (eval_res.num_examples, eval_res.metrics) for _, eval_res in results
        ])
        aggregated_loss = aggregated_metrics.get("loss", None)

        # Save to history
        if aggregated_loss is not None:
            self.eval_loss_history.append((server_round, aggregated_loss))
        if "accuracy" in aggregated_metrics:
            self.eval_acc_history.append((server_round, aggregated_metrics["accuracy"]))

        print(f"📊 Round {server_round} eval metrics: {aggregated_metrics}")

        # On last round, save eval JSON & plot
        if server_round == self.max_rounds:
            self.save_eval_history_json()
            self.plot_eval_convergence()
            self.model.save(self.export_path)
            print(f"✅ Saved Keras model to {self.export_path}")
            if self.tflite_path:
                export_to_tflite(self.model, self.tflite_path)
                print(f"✅ Exported to TFLite: {self.tflite_path}")

        # Must return two values
        return aggregated_loss, aggregated_metrics

    # Save & plot train history
    def save_train_history_json(self, path="fedrep_train_history_20.json"):
        data = {"loss": self.train_loss_history, "accuracy": self.train_acc_history}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"📝 Saved training history to {path}")

    def plot_train_convergence(self, save_path="fedrep_train_plot_20.png"):
        if not self.train_loss_history and not self.train_acc_history:
            print("⚠️ Nothing to plot for training.")
            return

        plt.figure(figsize=(10, 4))
        if self.train_loss_history:
            rounds, losses = zip(*self.train_loss_history)
            plt.subplot(1, 2, 1)
            plt.plot(rounds, losses, marker='o')
            plt.title("Train Loss over Rounds")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.grid(True)
        if self.train_acc_history:
            rounds, accs = zip(*self.train_acc_history)
            plt.subplot(1, 2, 2)
            plt.plot(rounds, accs, marker='o', color='green')
            plt.title("Train Accuracy over Rounds")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Saved training plot to {save_path}")
        plt.close()

    # Save & plot eval history
    def save_eval_history_json(self, path="fedrep_eval_history_20.json"):
        data = {"loss": self.eval_loss_history, "accuracy": self.eval_acc_history}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"📝 Saved evaluation history to {path}")

    def plot_eval_convergence(self, save_path="fedrep_eval_plot_20.png"):
        if not self.eval_loss_history and not self.eval_acc_history:
            print("⚠️ Nothing to plot for evaluation.")
            return

        plt.figure(figsize=(10, 4))
        if self.eval_loss_history:
            rounds, losses = zip(*self.eval_loss_history)
            plt.subplot(1, 2, 1)
            plt.plot(rounds, losses, marker='o')
            plt.title("Eval Loss over Rounds")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.grid(True)
        if self.eval_acc_history:
            rounds, accs = zip(*self.eval_acc_history)
            plt.subplot(1, 2, 2)
            plt.plot(rounds, accs, marker='o', color='green')
            plt.title("Eval Accuracy over Rounds")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Saved evaluation plot to {save_path}")
        plt.close()



def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", 10))
    model = build_fedrep_model()
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = SaveFedRep(
        model=model,
        export_path="final_model.keras",
        tflite_path="final_model.tflite",
        max_rounds=num_rounds,
        proximal_mu=0.01,
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)












# FEDPER



# import os
# import json
# import torch
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg
# from typing import Dict

# from sensorflow_model.task import ANN  # your PyTorch ANN class


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
#     return {"accuracy": float(weighted_result)}


# class SaveFedAvg(FedAvg):
#     def __init__(self, model, export_path="final_model.pt", max_rounds=10, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.export_path = export_path
#         self.max_rounds = max_rounds
#         self.loss_history = []
#         self.acc_history = []

#     def aggregate_fit(self, server_round, results, failures):
#         if not results:
#             return None, {}

#         try:
#             # Convert parameters to ndarrays
#             weights_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

#             # Simple average
#             aggregated_ndarrays = [np.mean(layer_group, axis=0) for layer_group in zip(*weights_ndarrays)]
#             aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

#             # Final round: update model and save
#             if server_round == self.max_rounds and aggregated_parameters is not None:
#                 self._set_model_weights(aggregated_ndarrays)
#                 torch.save(self.model.state_dict(), self.export_path)
#                 print(f"✅ Saved PyTorch model to {self.export_path}")


#         except Exception as e:
#             print(f"❌ Error during aggregation: {e}")
#             return None, {}

#         # Aggregate metrics
#         aggregated_metrics = self.aggregate_fit_metrics([
#             (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
#         ])

#         # Save history for convergence plot
#         if "loss" in aggregated_metrics:
#             self.loss_history.append((server_round, aggregated_metrics["loss"]))
#         if "accuracy" in aggregated_metrics:
#             self.acc_history.append((server_round, aggregated_metrics["accuracy"]))

#         if server_round == self.max_rounds:
#             self.plot_convergence()

#         return aggregated_parameters, aggregated_metrics

#     def _set_model_weights(self, ndarrays):
#         """Load averaged weights into PyTorch model"""
#         state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), ndarrays)}
#         self.model.load_state_dict(state_dict)

#     def aggregate_fit_metrics(self, results):
#         if not results:
#             return {}
#         total_examples = sum(num_examples for num_examples, _ in results)
#         metrics = {}
#         for num_examples, client_metrics in results:
#             for k, v in client_metrics.items():
#                 metrics[k] = metrics.get(k, 0.0) + v * (num_examples / total_examples)
#         return metrics


#     def plot_convergence(self, save_path="fedper_convergence_50rounds_plot.png"):
#         if not self.loss_history and not self.acc_history:
#             print("⚠️ No convergence history found. Cannot plot.")
#             return

#         plt.figure(figsize=(10, 4))

#         if self.loss_history:
#             rounds, values = zip(*self.loss_history)
#             plt.subplot(1, 2, 1)
#             plt.plot(rounds, values, marker='o')
#             plt.title("Loss over Rounds")
#             plt.xlabel("Round")
#             plt.ylabel("Loss")
#             plt.grid(True)

#         if self.acc_history:
#             rounds, values = zip(*self.acc_history)
#             plt.subplot(1, 2, 2)
#             plt.plot(rounds, values, marker='o', color='green')
#             plt.title("Accuracy over Rounds")
#             plt.xlabel("Round")
#             plt.ylabel("Accuracy")
#             plt.grid(True)

#         plt.tight_layout()
#         plt.savefig(save_path)
#         print(f"📈 Saved convergence plot to {save_path}")
#         plt.close()

#         with open("fedper_convergence_50rounds.json", "w") as f:
#             json.dump({"loss": self.loss_history, "accuracy": self.acc_history}, f)
#             print("📝 Saved convergence history to fedper_convergence_50rounds.json")


# def server_fn(context: Context) -> ServerAppComponents:
#     num_rounds = int(context.run_config["num-server-rounds"])

#     # Instantiate model (adjust input_dim if needed)
#     model = ANN(args=type("Args", (), {"input_dim": 20})(), name="FedModel")

#     # Convert initial model weights to Flower parameters
#     dummy_weights = [p.detach().cpu().numpy() for p in model.state_dict().values()]
#     initial_parameters = ndarrays_to_parameters(dummy_weights)

#     strategy = SaveFedAvg(
#         model=model,
#         export_path="final_model.pt",
#         max_rounds=num_rounds,
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_available_clients=2,
#         initial_parameters=initial_parameters,
#         fit_metrics_aggregation_fn=aggregate_fit_metrics,
#         evaluate_metrics_aggregation_fn=weighted_average,
#     )

#     config = ServerConfig(num_rounds=num_rounds)
#     return ServerAppComponents(strategy=strategy, config=config)


# app = ServerApp(server_fn=server_fn)
