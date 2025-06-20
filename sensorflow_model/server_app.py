from flwr.common import Context, ndarrays_to_parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from sensorflow_model.task import load_model


def weighted_average(metrics) -> dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_result = sum(
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ) / total_examples
    return {"accuracy": float(weighted_result)}



def server_fn(context: Context) -> ServerAppComponents:
    # Get number of training rounds from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model parameters
    model = load_model()
    parameters = ndarrays_to_parameters(model.get_weights())

    # Define federated averaging strategy
    from flwr.server.strategy import FedAvg

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create and run the Flower server app
app = ServerApp(server_fn=server_fn)
