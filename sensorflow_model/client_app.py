# from flwr.client import NumPyClient, ClientApp
# from flwr.common import Context

# from sensorflow_model.task import load_data, load_model, get_parameters, set_parameters



# class FlowerClient(NumPyClient):
#     def __init__(self, model, data, epochs, batch_size, verbose):
#         self.model = model
#         self.x_train, self.y_train, self.x_test, self.y_test = data
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.verbose = verbose

#     # 
#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)

#         history = self.model.fit(
#             self.x_train,
#             self.y_train,
#             epochs=config.get("local_epochs", 1),
#             batch_size=config.get("batch_size", 32),
#             verbose=0,
#         )

#         updated_weights = self.model.get_weights()

#         # ✅ Return accuracy and loss from the last epoch
#         metrics = {
#             "accuracy": float(history.history["accuracy"][-1]),
#             "loss": float(history.history["loss"][-1]),
#         }

#         return updated_weights, len(self.x_train), metrics
    
    
#     def evaluate(self, parameters, config):
#         set_parameters(self.model, parameters)
#         loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
#         return loss, len(self.x_test), {"accuracy": accuracy}



# def client_fn(context: Context):
#     # Load model
#     model = load_model()

#     # Get partition ID from Flower
#     partition_id = context.node_config["partition-id"]

#     # Load data for that client
#     data = load_data(partition_id)

#     # Get training config
#     epochs = context.run_config["local-epochs"]
#     batch_size = context.run_config["batch-size"]
#     verbose = context.run_config.get("verbose", 0)

#     return FlowerClient(model, data, epochs, batch_size, verbose).to_client()


# # Flower ClientApp
# app = ClientApp(
#     client_fn=client_fn,
# )
# from flwr.client import NumPyClient, ClientApp
# from flwr.common import Context

# from sensorflow_model.task import load_data, load_model, get_parameters, set_parameters

# import numpy as np


# class FlowerClient(NumPyClient):
#     def __init__(self, model, data, epochs, batch_size, verbose):
#         self.model = model
#         self.x_train, self.y_train, self.x_test, self.y_test = data
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.verbose = verbose
#         self.mask = None  # Store the client's masking vector

#     def generate_mask(self, weights):
#         """Generate random Gaussian mask matching weight shapes."""
#         return [np.random.normal(loc=0.0, scale=0.01, size=w.shape).astype(np.float32) for w in weights]

#     def apply_mask(self, weights, mask):
#         """Add mask to weights element-wise."""
#         return [w + m for w, m in zip(weights, mask)]

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)

#         history = self.model.fit(
#             self.x_train,
#             self.y_train,
#             epochs=config.get("local_epochs", self.epochs),
#             batch_size=config.get("batch_size", self.batch_size),
#             verbose=self.verbose,
#         )

#         updated_weights = self.model.get_weights()

#         # Step 1: Generate and apply mask
#         self.mask = self.generate_mask(updated_weights)
#         masked_weights = self.apply_mask(updated_weights, self.mask)

#         # Step 2: Return masked weights and metrics
#         metrics = {
#             "accuracy": float(history.history["accuracy"][-1]),
#             "loss": float(history.history["loss"][-1]),
#         }

#         return masked_weights, len(self.x_train), metrics

#     def evaluate(self, parameters, config):
#         set_parameters(self.model, parameters)
#         loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
#         return loss, len(self.x_test), {"accuracy": accuracy}


# def client_fn(context: Context):
#     model = load_model()
#     partition_id = context.node_config["partition-id"]
#     data = load_data(partition_id)

#     epochs = context.run_config["local-epochs"]
#     batch_size = context.run_config["batch-size"]
#     verbose = context.run_config.get("verbose", 0)

#     return FlowerClient(model, data, epochs, batch_size, verbose).to_client()


# # Flower ClientApp
# app = ClientApp(
#     client_fn=client_fn,
# )
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from sensorflow_model.task import load_data, load_model, get_parameters, set_parameters

import numpy as np


class FlowerClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        # Dual-input: [x_watch, x_ear], y
        [self.x_train_watch, self.x_train_ear], self.y_train, [self.x_test_watch, self.x_test_ear], self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.mask = None  # Store the client's masking vector

    def generate_mask(self, weights):
        """Generate random Gaussian mask matching weight shapes."""
        return [np.random.normal(loc=0.0, scale=0.01, size=w.shape).astype(np.float32) for w in weights]

    def apply_mask(self, weights, mask):
        """Add mask to weights element-wise."""
        return [w + m for w, m in zip(weights, mask)]

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        history = self.model.fit(
            [self.x_train_watch, self.x_train_ear],
            self.y_train,
            epochs=config.get("local_epochs", self.epochs),
            batch_size=config.get("batch_size", self.batch_size),
            verbose=self.verbose,
        )

        updated_weights = self.model.get_weights()

        # Step 1: Generate and apply mask
        self.mask = self.generate_mask(updated_weights)
        masked_weights = self.apply_mask(updated_weights, self.mask)

        # Step 2: Return masked weights and metrics
        metrics = {
            "accuracy": float(history.history["accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
        }

        return masked_weights, len(self.y_train), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = self.model.evaluate([self.x_test_watch, self.x_test_ear], self.y_test, verbose=0)
        return loss, len(self.y_test), {"accuracy": accuracy}


def client_fn(context: Context):
    model = load_model()
    partition_id = context.node_config["partition-id"]
    data = load_data(partition_id)

    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", 0)

    return FlowerClient(model, data, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
