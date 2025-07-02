from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
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



def weighted_average(metrics) -> dict[str, Scalar]:
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






# from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Scalar
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg
# from sensorflow_model.task import load_model, get_parameters, set_parameters, export_to_tflite

# def weighted_average(metrics) -> dict[str, Scalar]:
#     total_examples = sum(num_examples for num_examples, _ in metrics)
#     weighted_result = sum(
#         num_examples * metric["accuracy"]  # or "loss"
#         for num_examples, metric in metrics
#     ) / total_examples
#     return {"accuracy": float(weighted_result)}



# def weighted_average_fit(metrics) -> dict[str, Scalar]:
#     total_examples = sum(num_examples for num_examples, _ in metrics)
#     weighted_result = sum(
#         num_examples * metric["accuracy"]  # or "loss"
#         for num_examples, metric in metrics
#     ) / total_examples
#     return {"accuracy": float(weighted_result)}




# def server_fn(context: Context) -> ServerAppComponents:
#     # Get number of training rounds from config
#     num_rounds = context.run_config["num-server-rounds"]

#     # Initialize global model parameters
#     model = load_model()
#     weights = get_parameters(model)
#     parameters = ndarrays_to_parameters(weights)
    


#     strategy = FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_available_clients=2,
#         initial_parameters=parameters,
#         evaluate_metrics_aggregation_fn=weighted_average,
#         # fit_metrics_aggregation_fn=weighted_average_fit,
#     )

#     config = ServerConfig(num_rounds=int(num_rounds))

#     return ServerAppComponents(strategy=strategy, config=config)


# # Create and run the Flower server app
# app = ServerApp(server_fn=server_fn)
