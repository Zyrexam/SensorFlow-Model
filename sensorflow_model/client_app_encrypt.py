import numpy as np
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from sensorflow_model.task import load_data, load_model, get_parameters, set_parameters

# ------------- SECURE AGGREGATION UTILS -----------------
def split_weights_shares(weights, num_clients, client_id):
    """
    Create additive shares for model weights.
    In this simulation, each client just keeps its own share,
    which is its actual model update minus random noise.
    """
    shares = []
    for w in weights:
        share_shape = w.shape
        # Generate n-1 random shares for other clients (simulate mask)
        rnd_shares = [np.random.normal(0, 0.01, size=share_shape) for _ in range(num_clients - 1)]
        my_share = w - sum(rnd_shares)
        # In simulation: only keep/send my_share rather than exchanging with all peers
        shares.append(my_share)
    return shares

# ------------- CLIENT DEFINITION ------------------------

class FlowerClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose, partition_id, num_clients=6):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.partition_id = partition_id
        self.num_clients = num_clients

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=config.get("local_epochs", 1),
            batch_size=config.get("batch_size", 32),
            verbose=0,
        )

        updated_weights = self.model.get_weights()
        # --- Secure Aggregation: send only the “share”, not the raw weights ---
        shares = split_weights_shares(updated_weights, num_clients=self.num_clients, client_id=self.partition_id)

        metrics = {
            "accuracy": float(history.history["accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
        }

        return shares, len(self.x_train), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}

# --------- CLIENT APP CONSTRUCTOR -------------

def client_fn(context: Context):
    # Load model
    model = load_model()

    # Get partition ID from Flower (0-based)
    partition_id = context.node_config["partition-id"]

    # Load data for that client
    data = load_data(partition_id)

    # Get training config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", 0)
    num_clients = 6  # update if number of participating clients changes

    return FlowerClient(model, data, epochs, batch_size, verbose, partition_id, num_clients).to_client()

# -------------- FLWR APPLICATION ENTRANCE POINT ---------------

app = ClientApp(
    client_fn=client_fn,
)
