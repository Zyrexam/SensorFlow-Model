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




#with MASKING

# from flwr.client import NumPyClient, ClientApp
# from flwr.common import Context

# from sensorflow_model.task import load_data, load_model, get_parameters, set_parameters

# import numpy as np


# class FlowerClient(NumPyClient):
#     def __init__(self, model, data, epochs, batch_size, verbose):
#         self.model = model
#         # Dual-input: [x_watch, x_ear], y
#         [self.x_train_watch, self.x_train_ear], self.y_train, [self.x_test_watch, self.x_test_ear], self.y_test = data
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
#             [self.x_train_watch, self.x_train_ear],
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

#         return masked_weights, len(self.y_train), metrics

#     def evaluate(self, parameters, config):
#         set_parameters(self.model, parameters)
#         loss, accuracy = self.model.evaluate([self.x_test_watch, self.x_test_ear], self.y_test, verbose=0)
#         return loss, len(self.y_test), {"accuracy": accuracy}


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







# FEDPROX

# from flwr.client import NumPyClient, ClientApp
# from flwr.common import Context
# from sensorflow_model.task import load_data, load_model, get_parameters, set_parameters

# import numpy as np
# import tensorflow as tf

# class FlowerClient(NumPyClient):
#     def __init__(self, model, data, epochs, batch_size, verbose, proximal_mu=0.01):
#         self.model = model
#         [self.x_train_watch, self.x_train_ear], self.y_train, \
#         [self.x_test_watch, self.x_test_ear], self.y_test = data
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.verbose = verbose
#         self.proximal_mu = proximal_mu

#     def fit(self, parameters, config):
#         # Set global parameters
#         self.model.set_weights(parameters)

#         # Convert global parameters to tensors
#         global_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in parameters]

#         optimizer = tf.keras.optimizers.Adam()
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


#         batch_size = int(config.get("batch_size", self.batch_size))
        
#         local_epochs = int(config.get("local_epochs", config.get("local-epochs", self.epochs)))

#         train_dataset = tf.data.Dataset.from_tensor_slices(
#             ((self.x_train_watch, self.x_train_ear), self.y_train)
#         ).batch(batch_size)

#         for epoch in range(local_epochs):
#             epoch_loss = []
#             for (batch_x_watch, batch_x_ear), batch_y in train_dataset:
#                 with tf.GradientTape() as tape:
#                     logits = self.model([batch_x_watch, batch_x_ear], training=True)
#                     loss = loss_fn(batch_y, logits)

#                     # Get current local weights as tensors
#                     local_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in self.model.get_weights()]

#                     # Compute FedProx proximal term
#                     prox_term = tf.constant(0.0, dtype=tf.float32)
#                     for lw, gw in zip(local_weights, global_weights):
#                         prox_term += tf.nn.l2_loss(lw - gw)
#                     loss += (self.proximal_mu / 2) * prox_term

#                 grads = tape.gradient(loss, self.model.trainable_variables)
#                 optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
#                 epoch_loss.append(float(loss.numpy()))

#             if self.verbose:
#                 print(f"Epoch {epoch+1}: mean loss = {np.mean(epoch_loss):.4f}")

#         # Compute final training accuracy
#         train_logits = self.model.predict([self.x_train_watch, self.x_train_ear], verbose=0)
#         train_preds = np.argmax(train_logits, axis=1)
#         train_acc = np.mean(train_preds == self.y_train)

#         metrics = {
#             "accuracy": float(train_acc),
#             "loss": float(np.mean(epoch_loss)),
#         }

#         updated_weights = self.model.get_weights()
#         return updated_weights, len(self.y_train), metrics

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, accuracy = self.model.evaluate([self.x_test_watch, self.x_test_ear], self.y_test, verbose=0)
#         return float(loss), len(self.y_test), {"accuracy": float(accuracy)}





# def client_fn(context: Context):
#     model = load_model()
#     partition_id = context.node_config["partition-id"]
#     data = load_data(partition_id)

#     epochs = context.run_config["local-epochs"]
#     batch_size = context.run_config["batch-size"]
#     verbose = context.run_config.get("verbose", 0)
#     proximal_mu = float(context.run_config.get("proximal-mu", 0.01))


#     return FlowerClient(model, data, epochs, batch_size, verbose, proximal_mu).to_client()


# app = ClientApp(
#     client_fn=client_fn,
# )


# # FEDPER


# import numpy as np
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from torch import nn, optim

# from flwr.client import NumPyClient, ClientApp
# from flwr.common import Context

# from sensorflow_model.task import load_data, ANN, get_parameters, set_parameters


# # ------------- SECURE AGGREGATION UTILS -----------------
# def split_weights_shares(weights, num_clients, client_id):
#     """
#     Create additive shares for model weights.
#     """
#     shares = []
#     for w in weights:
#         share_shape = w.shape
#         rnd_shares = [np.random.normal(0, 0.01, size=share_shape) for _ in range(num_clients - 1)]
#         my_share = w - sum(rnd_shares)
#         shares.append(my_share.astype(np.float32))
#     return shares


# # ------------- CLIENT DEFINITION ------------------------
# class FlowerClient(NumPyClient):
#     def __init__(self, model, data, epochs, batch_size, verbose, partition_id, num_clients=6):
#         self.model = model
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.verbose = verbose
#         self.partition_id = partition_id
#         self.num_clients = num_clients

#         # Convert numpy data to torch tensors
#         (x_train_watch, x_train_ear), y_train, (x_test_watch, x_test_ear), y_test = data
#         self.x_train_watch = torch.tensor(x_train_watch, dtype=torch.float32)
#         self.x_train_ear = torch.tensor(x_train_ear, dtype=torch.float32)
#         self.y_train = torch.tensor(y_train, dtype=torch.long)

#         self.x_test_watch = torch.tensor(x_test_watch, dtype=torch.float32)
#         self.x_test_ear = torch.tensor(x_test_ear, dtype=torch.float32)
#         self.y_test = torch.tensor(y_test, dtype=torch.long)

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def fit(self, parameters, config):
#         # Set model parameters
#         state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
#         self.model.load_state_dict(state_dict)

#         self.model.train()
#         optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         criterion = nn.CrossEntropyLoss()

#         # Combine watch & ear inputs into dataset
#         dataset = TensorDataset(self.x_train_watch, self.x_train_ear, self.y_train)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(self.epochs):
#             epoch_loss, correct, total = 0.0, 0, 0
#             for watch, ear, labels in loader:
#                 watch, ear, labels = watch.to(self.device), ear.to(self.device), labels.to(self.device)

#                 optimizer.zero_grad()
#                 outputs = self.model((watch, ear))
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 epoch_loss += loss.item() * labels.size(0)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)

          
#         avg_loss = float(epoch_loss) / int(total)
#         avg_acc = float(correct) / int(total)


#         # Get updated model weights as NumPy arrays
#         updated_weights = [v.cpu().detach().numpy() for v in self.model.state_dict().values()]

#         # Secure aggregation: only send share
#         shares = split_weights_shares(updated_weights, num_clients=self.num_clients, client_id=self.partition_id)

#         metrics = {"accuracy": float(avg_acc), "loss": float(avg_loss)}
#         return shares, total, metrics

#     def evaluate(self, parameters, config):
#         # Set model parameters
#         state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
#         self.model.load_state_dict(state_dict)

#         self.model.eval()
#         criterion = nn.CrossEntropyLoss()

#         dataset = TensorDataset(self.x_test_watch, self.x_test_ear, self.y_test)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

#         total_loss, correct, total = 0.0, 0, 0
#         with torch.no_grad():
#             for watch, ear, labels in loader:
#                 watch, ear, labels = watch.to(self.device), ear.to(self.device), labels.to(self.device)

#                 outputs = self.model((watch, ear))
#                 loss = criterion(outputs, labels)

#                 total_loss += loss.item() * labels.size(0)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)

#         avg_loss = total_loss / total
#         avg_acc = correct / total
#         return float(avg_loss), total, {"accuracy": float(avg_acc)}


# # --------- CLIENT APP CONSTRUCTOR -------------
# def client_fn(context: Context):
#     # Partition ID from Flower
#     partition_id = context.node_config["partition-id"]

#     # Load PyTorch ANN model
#     model = ANN(args=None, name=f"client_{partition_id}")

#     # Load local partition data
#     data = load_data(partition_id)

#     # Training config
#     epochs = context.run_config["local-epochs"]
#     batch_size = context.run_config["batch-size"]
#     verbose = context.run_config.get("verbose", 0)
#     num_clients = 6

#     return FlowerClient(model, data, epochs, batch_size, verbose, partition_id, num_clients).to_client()


# # -------------- FLWR APPLICATION ENTRANCE POINT ---------------
# app = ClientApp(client_fn=client_fn)











#FEDREP 


from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from sensorflow_model.task import load_data, build_fedrep_model
import numpy as np
import tensorflow as tf
class FedRepClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        # Split layers
        self.backbone_layers = [l for l in self.model.layers if "dense" in l.name and l.name != "head"]
        self.head_layers = [self.model.get_layer("head")]

    def fit(self, parameters, config):
        # Load backbone parameters into backbone layers
        full_weights = self.model.get_weights()
        num_backbone = len(parameters)
        full_weights[:num_backbone] = parameters
        self.model.set_weights(full_weights)

        # Step 1: Train head (local personalization)
        for l in self.backbone_layers:
            l.trainable = False
        for l in self.head_layers:
            l.trainable = True

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_head = self.model.fit(
            [self.x_train[0], self.x_train[1]], self.y_train,
            epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
        )

        # Step 2: Federated step: train backbone
        for l in self.backbone_layers:
            l.trainable = True
        for l in self.head_layers:
            l.trainable = False

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_backbone = self.model.fit(
            [self.x_train[0], self.x_train[1]], self.y_train,
            epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
        )

        # Get final weights (only backbone part to send back)
        new_weights = self.model.get_weights()

        # Collect local metrics (last epoch)
        final_train_loss = float(history_backbone.history["loss"][-1])
        final_train_acc = float(history_backbone.history["accuracy"][-1])

        # Return backbone weights + number of samples + local training metrics
        return new_weights[:num_backbone], len(self.y_train), {
            "loss": final_train_loss,
            "accuracy": final_train_acc
        }

    def evaluate(self, parameters, config):
        # Load backbone params into backbone layers
        full_weights = self.model.get_weights()
        num_backbone = len(parameters)
        full_weights[:num_backbone] = parameters
        self.model.set_weights(full_weights)

        # Evaluate the personalized model: backbone + local head
        for l in self.backbone_layers:
            l.trainable = False
        for l in self.head_layers:
            l.trainable = True

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        loss, accuracy = self.model.evaluate([self.x_test[0], self.x_test[1]], self.y_test, verbose=0)
        return float(loss), len(self.y_test), {"accuracy": float(accuracy)}


# client_fn creates a FedRepClient
def client_fn(context: Context):
    model = build_fedrep_model()
    partition_id = context.node_config["partition-id"]
    data = load_data(partition_id)

    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", 0)

    return FedRepClient(model, data, epochs, batch_size, verbose).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)
