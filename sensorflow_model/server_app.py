from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from typing import Dict
from flwr.server.strategy import FedAvg
from sensorflow_model.task import (
    load_model,
    get_parameters,
    set_parameters,
    export_to_tflite, 
)

import json



ACTIVITY_CLASSES = {
    0: "Sitting + Typing on Desk",
    1: "Sitting + Taking Notes", 
    2: "Standing + Writing on Whiteboard",
    3: "Standing + Erasing Whiteboard",
    4: "Sitting + Talking + Waving Hands",
    5: "Standing + Talking + Waving Hands",
    6: "Sitting + HeadNodding",
    7: "Sitting + Drinking Water",
    8: "Sitting + Drinking Coffee",
    9: "Standing + Drinking Water",
    10: "Standing + Drinking Coffee",
    11: "Scrolling on Phone",
}



def weighted_average(metrics) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_result = sum(
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ) / total_examples
    return {"accuracy": float(weighted_result)}



class SaveFedAvg(FedAvg):
    def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.export_path = export_path
        self.tflite_path = tflite_path
        self.max_rounds = max_rounds

    def aggregate_fit(self, server_round, results, failures):
        aggregated_result = super().aggregate_fit(server_round, results, failures)

        if aggregated_result is not None:
            aggregated_parameters, aggregated_metrics = aggregated_result

            #  Set weights and save at final round
            if server_round == self.max_rounds and aggregated_parameters is not None:
                weights = parameters_to_ndarrays(aggregated_parameters)
                self.model.set_weights(weights)

                #  Save in Keras format
                self.model.save(self.export_path)
                print(f" Saved Keras model to {self.export_path}")

                if self.tflite_path:
                    export_to_tflite(self.model, self.tflite_path)
                    print(f" Exported model to TFLite: {self.tflite_path}")

            return aggregated_parameters, aggregated_metrics

        return None


# class SaveFedAvg(FedAvg):
#     def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.export_path = export_path
#         self.tflite_path = tflite_path
#         self.max_rounds = max_rounds

#         self.metrics_over_rounds = {
#             "round": [],
#             "train_loss": [],
#             "val_loss": [],
#             "train_accuracy": [],
#             "val_accuracy": [],
#         }


#     def aggregate_fit(self, server_round, results, failures):
#         aggregated_result = super().aggregate_fit(server_round, results, failures)

#         if aggregated_result is not None:
#             aggregated_parameters, aggregated_metrics = aggregated_result

#             # Collect average metrics from clients
#             total_examples = sum(fit_res.num_examples for _, fit_res in results)
#             avg_metrics = {
#                 "train_loss": 0.0,
#                 "val_loss": 0.0,
#                 "train_accuracy": 0.0,
#                 "val_accuracy": 0.0,
#             }

#             for _, fit_res in results:
#                 metrics = fit_res.metrics if hasattr(fit_res, "metrics") else {}
#                 num_examples = fit_res.num_examples if hasattr(fit_res, "num_examples") else 0
#                 for key in avg_metrics:
#                     if key in metrics:
#                         avg_metrics[key] += float(metrics[key]) * num_examples

#             for key in avg_metrics:
#                 avg_metrics[key] /= total_examples

#             # Save to in-memory metrics tracker
#             self.metrics_over_rounds["round"].append(server_round)
#             self.metrics_over_rounds["train_loss"].append(avg_metrics["train_loss"])
#             self.metrics_over_rounds["val_loss"].append(avg_metrics["val_loss"])
#             self.metrics_over_rounds["train_accuracy"].append(avg_metrics["train_accuracy"])
#             self.metrics_over_rounds["val_accuracy"].append(avg_metrics["val_accuracy"])

#             # Save final model
#             if server_round == self.max_rounds and aggregated_parameters is not None:
#                 weights = parameters_to_ndarrays(aggregated_parameters)
#                 self.model.set_weights(weights)

#                 self.model.save(self.export_path)
#                 print(f"✅ Saved Keras model to {self.export_path}")

#                 if self.tflite_path:
#                     export_to_tflite(self.model, self.tflite_path)
#                     print(f"✅ Exported model to TFLite: {self.tflite_path}")

#                 # Save metrics to CSV
#                 import pandas as pd
#                 df = pd.DataFrame(self.metrics_over_rounds)
#                 df.to_csv("metrics_over_rounds.csv", index=False)
#                 print("📊 Saved metrics to metrics_over_rounds.csv")

#             return aggregated_parameters, aggregated_metrics

#         return None


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
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
