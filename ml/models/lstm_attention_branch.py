"""
Branche 2 : LSTM + attention sur la SÉQUENCE des dernières transactions de l'utilisateur.

Entrée : (batch, SEQ_LEN, n_features), la transaction à scorer en dernière position,
les positions vides (historique trop court) étant masquées. L'attention indique quelles
transactions passées ont compté dans la décision (utile pour l'explicabilité).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rnn_output, mask=None):
        scores = self.attention(rnn_output).squeeze(-1)          # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * rnn_output, dim=1)
        return context, weights


class LSTMAttentionBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionBlock(hidden_dim)
        # contexte (historique pondéré) + état de la transaction courante
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x, mask=None, return_attention: bool = False):
        rnn_out, _ = self.lstm(x)
        context, weights = self.attention(rnn_out, mask)
        logits = self.head(torch.cat([context, rnn_out[:, -1, :]], dim=1)).squeeze(-1)
        return (logits, weights) if return_attention else logits


def gather_sequences(X: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Construit le lot (B, L, F) et son masque à partir des indices de séquence."""
    idx = seq_idx[rows]
    mask = idx >= 0
    seq = X[np.where(mask, idx, 0)]
    seq[~mask] = 0.0
    return torch.from_numpy(seq), torch.from_numpy(mask)


@torch.no_grad()
def lstm_logit(model: LSTMAttentionBranch, X: np.ndarray, seq_idx: np.ndarray, rows: np.ndarray,
               batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(rows), batch_size):
        xb, mb = gather_sequences(X, seq_idx, rows[s:s + batch_size])
        out.append(model(xb, mb).numpy())
    return np.concatenate(out).astype(np.float32) if out else np.empty(0, dtype=np.float32)


def train_lstm_branch(X: np.ndarray, y: np.ndarray, seq_idx: np.ndarray, fit_rows: np.ndarray,
                      es_rows: np.ndarray, epochs: int = 8, batch_size: int = 512, lr: float = 1e-3,
                      neg_sample_rate: float = 0.3, patience: int = 2, hidden_dim: int = 64,
                      seed: int = 42, log=print) -> LSTMAttentionBranch:
    """Entraînement avec sous-échantillonnage des transactions légitimes à chaque époque
    (toutes les fraudes + neg_sample_rate des légitimes, tirées à nouveau à chaque époque)
    et arrêt précoce sur la PR-AUC de la tranche early_stop."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LSTMAttentionBranch(X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    pos_rows, neg_rows = fit_rows[y[fit_rows] == 1], fit_rows[y[fit_rows] == 0]
    ratio = len(neg_rows) * neg_sample_rate / max(len(pos_rows), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt(ratio), dtype=torch.float32))

    best_ap, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        n_neg = int(len(neg_rows) * neg_sample_rate)
        epoch_rows = np.concatenate([pos_rows, rng.choice(neg_rows, n_neg, replace=False)])
        rng.shuffle(epoch_rows)
        total = 0.0
        for s in range(0, len(epoch_rows), batch_size):
            b = epoch_rows[s:s + batch_size]
            xb, mb = gather_sequences(X, seq_idx, b)
            yb = torch.from_numpy(y[b].astype(np.float32))
            optimizer.zero_grad()
            loss = criterion(model(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(b)

        ap = average_precision_score(y[es_rows], lstm_logit(model, X, seq_idx, es_rows))
        log(f"    époque {epoch}/{epochs}  perte={total / len(epoch_rows):.4f}  PR-AUC(early_stop)={ap:.4f}")
        if ap > best_ap + 1e-4:
            best_ap, bad_epochs = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log("    arrêt précoce")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model
