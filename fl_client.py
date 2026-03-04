"""
fl_client.py — Flower federated learning client
Each client wraps a local hospital data shard + ADRNet model.
"""

import numpy as np
import flwr as fl
from model import ADRNet, train_local, get_weights, set_weights, evaluate

INPUT_DIM_WITH_CATE = 26 + 5   # base features + 5 CATE features


class FPSClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray):
        self.client_id = client_id
        self.X_train   = X_train
        self.y_train   = y_train
        self.X_test    = X_test
        self.y_test    = y_test
        self.model     = ADRNet(input_dim=X_train.shape[1])

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        result = train_local(self.model, self.X_train, self.y_train,
                             epochs=config.get("epochs", 5))
        return get_weights(self.model), len(self.X_train), {"loss": result["loss"]}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        metrics = evaluate(self.model, self.X_test, self.y_test)
        # Flower expects (loss, n_samples, metrics_dict)
        # Use 1 - AUROC as a proxy loss for aggregation
        loss = float(1.0 - metrics["auroc"])
        return loss, len(self.X_test), metrics