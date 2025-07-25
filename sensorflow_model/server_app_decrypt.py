import os
import json
import numpy as np
from typing import Dict

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from sensorflow_model.personalize import evaluate_personalization_on_clients
from sensorflow_model.task import (
    load_model,
    get_parameters,
    set_parameters,
    export_to_tflite,
)


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
    return {"accuracy": float(weighted_result)}

# --- SECURE AGGREGATION CORE ----
def aggregate_shares(client_shares_list):
    """
    Receives list of shares (one from each client).
    Each share is a list of np.ndarrays.
    Returns the average of all clients' weights (FedAvg).
    """
    num_clients = len(client_shares_list)
    agg = []
    for weights in zip(*client_shares_list):
        summed = np.sum(np.stack(weights), axis=0)
        agg.append(summed / num_clients)  # mean, as in FedAvg
    return agg

# --- FLOWER STRATEGY OVERRIDE ----
class SaveFedAvgSecure(FedAvg):
    def __init__(self, model, export_path="final_model.keras", tflite_path=None, max_rounds=10, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.export_path = export_path
        self.tflite_path = tflite_path
        self.max_rounds = max_rounds

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None

        # Securely aggregate shares
        weights_received = [parameters_to_ndarrays(fit_res.parameters) for fit_res, _, _ in results]
        aggregated_ndarrays = aggregate_shares(weights_received)
        aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)

        # Compute average metrics
        total_examples = sum(fit_res.num_examples for fit_res, _, _ in results)
        avg_accuracy = sum(fit_res.num_examples * fit_res.metrics.get("accuracy", 0.0)
                           for fit_res, _, _ in results) / total_examples
        avg_loss = sum(fit_res.num_examples * fit_res.metrics.get("loss", 0.0)
                       for fit_res, _, _ in results) / total_examples
        aggregated_metrics = {"accuracy": avg_accuracy, "loss": avg_loss}

        # Save model and optionally export as .tflite at the end
        if server_round == self.max_rounds and aggregated_params is not None:
            weights = parameters_to_ndarrays(aggregated_params)
            self.model.set_weights(weights)
            self.model.save(self.export_path)
            print(f" Saved Keras model to {self.export_path}")
            if self.tflite_path:
                export_to_tflite(self.model, self.tflite_path)
                print(f" Exported model to TFLite: {self.tflite_path}")

            # Post-training personalization evaluation
            evaluate_personalization_on_clients(self.model, data_folder="myData", k_values=[5, 10, 20])

        return aggregated_params, aggregated_metrics

def save_normalization_stats(mean, std, path="normalization.json"):
    with open(path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"Saved normalization stats to {path}")

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config["num-server-rounds"])
    model = load_model()
    weights = get_parameters(model)
    parameters = ndarrays_to_parameters(weights)
    
    strategy = SaveFedAvgSecure(
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
