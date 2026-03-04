"""
fl_client.py — Federated Learning simulation (FedProx)
=======================================================
Custom in-process FedProx loop — no deprecated Flower API needed.

Why not flwr.simulation.start_simulation()?
  That API is deprecated in Flower >= 1.8. The replacement requires a
  CLI-driven project structure that doesn't fit an in-process pipeline.
  A manual FedProx loop gives direct weight access for Step 7 attribution
  and avoids the deprecation warning entirely.

FedProx objective (Li et al. 2020):
  min_w  F_k(w)  +  (mu/2) * ||w - w_global||^2
  mu=0 -> plain FedAvg, mu>0 -> stabilises non-IID divergence

Class-imbalance fix:
  ADR rate ~45% — using weighted BCE so the model doesn't collapse to
  predicting all-negative during FL aggregation (root cause of AUROC < 0.5).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import ADRNet, get_weights, set_weights


def train_fedprox(model, global_weights, X, y,
                  epochs=5, lr=1e-3, proximal_mu=0.1):
    """Local FedProx training with weighted BCE and gradient clipping."""
    model.train()

    n_pos      = max(float(y.sum()), 1.0)
    n_neg      = max(float(len(y) - y.sum()), 1.0)
    pos_w      = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    loader    = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce       = nn.BCELoss(reduction="none")
    g_tensors = [torch.tensor(w, dtype=torch.float32) for w in global_weights]

    total_loss = 0.0
    for _ in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            prob, _ = model(xb)

            # Weighted BCE: upweight positive (ADR=1) samples
            weights = torch.where(yb == 1,
                                   pos_w.expand_as(yb),
                                   torch.ones_like(yb))
            loss = (bce(prob, yb) * weights).mean()

            # FedProx proximal term
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), g_tensors))
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        total_loss = ep_loss / len(loader)

    return {"loss": total_loss}


def fedavg_aggregate(client_weights, client_sizes):
    """Weighted average of client weights by local dataset size."""
    total = sum(client_sizes)
    return [
        sum(w[i] * (n / total) for w, n in zip(client_weights, client_sizes))
        for i in range(len(client_weights[0]))
    ]


def run_federated(splits, input_dim, fl_rounds=20,
                  local_epochs=5, proximal_mu=0.1, evaluate_fn=None):
    """
    Run FedProx simulation.

    Parameters
    ----------
    splits       : list of (X_tr, y_tr, X_te, y_te) per hospital
    input_dim    : feature vector size
    fl_rounds    : communication rounds
    local_epochs : local epochs per client per round
    proximal_mu  : FedProx mu (0 = plain FedAvg)
    evaluate_fn  : callable(round_num, global_model) -> metrics dict

    Returns (round_metrics, global_model)
    """
    global_model   = ADRNet(input_dim=input_dim)
    global_weights = get_weights(global_model)
    round_metrics  = []

    for rnd in range(1, fl_rounds + 1):
        client_weights, client_sizes = [], []

        for X_tr, y_tr, *_ in splits:
            local = ADRNet(input_dim=input_dim)
            set_weights(local, copy.deepcopy(global_weights))
            train_fedprox(local, global_weights, X_tr, y_tr,
                          epochs=local_epochs, proximal_mu=proximal_mu)
            client_weights.append(get_weights(local))
            client_sizes.append(len(X_tr))

        global_weights = fedavg_aggregate(client_weights, client_sizes)
        set_weights(global_model, global_weights)

        if evaluate_fn is not None:
            metrics = evaluate_fn(rnd, global_model)
            round_metrics.append((rnd, metrics))

    return round_metrics, global_model