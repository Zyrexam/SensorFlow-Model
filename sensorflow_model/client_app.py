from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from task import load_data, load_model, get_parameters, set_parameters


class FlowerClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return get_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}



def client_fn(context: Context):
    # Load model
    model = load_model()

    # Get partition ID from Flower
    partition_id = context.node_config["partition-id"]

    # Load data for that client
    data = load_data(partition_id)

    # Get training config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", 0)

    return FlowerClient(model, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
