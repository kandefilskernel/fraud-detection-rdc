"""
Branche 3 : autoencodeur (détection d'anomalies non supervisée).

Entraîné UNIQUEMENT sur des transactions légitimes : il apprend à reconstruire le
comportement normal. Une erreur de reconstruction élevée signale une transaction atypique,
y compris pour des typologies de fraude jamais vues à l'entraînement.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AutoencoderBranch(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=1)


@torch.no_grad()
def ae_score(model: AutoencoderBranch, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """log de l'erreur de reconstruction (distribution moins asymétrique)."""
    model.eval()
    out = [model.reconstruction_error(torch.from_numpy(X[s:s + batch_size])).numpy()
           for s in range(0, len(X), batch_size)]
    return np.log(np.concatenate(out) + 1e-6).astype(np.float32)


def train_autoencoder_branch(X_fit_legit: np.ndarray, X_es_legit: np.ndarray, epochs: int = 30,
                             batch_size: int = 1024, lr: float = 1e-3, patience: int = 3,
                             seed: int = 42, log=print) -> AutoencoderBranch:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AutoencoderBranch(X_fit_legit.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_es_t = torch.from_numpy(X_es_legit)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X_fit_legit))
        for s in range(0, len(order), batch_size):
            xb = torch.from_numpy(X_fit_legit[order[s:s + batch_size]])
            optimizer.zero_grad()
            loss = torch.mean((model(xb) - xb) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            es_loss = model.reconstruction_error(X_es_t).mean().item()
        if epoch == 1 or epoch % 5 == 0:
            log(f"    époque {epoch}/{epochs}  erreur(early_stop, légitimes)={es_loss:.4f}")
        if es_loss < best - 1e-5:
            best, bad = es_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"    arrêt précoce à l'époque {epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model
