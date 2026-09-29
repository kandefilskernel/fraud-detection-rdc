"""
Forêt aléatoire « aplatie » pour le temps réel.

scikit-learn parcourt les arbres un par un en Python : ~0,2 ms par arbre, soit ~40 ms pour
200 arbres et une seule transaction, trop lent pour le budget temps réel. Ici tous les nœuds
de tous les arbres sont rangés dans quelques tableaux numpy, et les 200 arbres descendent
EN MÊME TEMPS, un niveau par itération : la boucle Python ne dépend que de la profondeur.

Explication par variable (méthode de Saabas, additive et exacte) : le long du chemin suivi
dans chaque arbre, chaque séparation déplace la probabilité de fraude de p(parent) à
p(enfant) ; cet écart est attribué à la variable qui sépare. Pour chaque transaction :

    P(fraude) = biais + Σ contributions(variable)      (moyenne sur les arbres)

Contrairement à TreeSHAP, cette attribution n'est pas « cohérente » au sens de Lundberg (2017),
mais elle est exacte (la somme redonne la prédiction), rapide et sans dépendance.
"""
from __future__ import annotations

import numpy as np


class FlatForest:
    def __init__(self, rf):
        trees = [est.tree_ for est in rf.estimators_]
        sizes = np.array([t.node_count for t in trees])
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        feat, thr, left, right, prob = [], [], [], [], []
        for t, off in zip(trees, offsets):
            leaf = t.children_left == -1
            idx = np.arange(t.node_count)
            # une feuille pointe vers elle-même : les arbres moins profonds « attendent » les autres
            left.append(np.where(leaf, idx, t.children_left) + off)
            right.append(np.where(leaf, idx, t.children_right) + off)
            feat.append(np.where(leaf, 0, t.feature))
            thr.append(np.where(leaf, np.inf, t.threshold))
            v = t.value[:, 0, :]
            prob.append(v[:, 1] / np.maximum(v.sum(axis=1), 1e-12))   # probabilité de fraude du nœud
        self.feature = np.concatenate(feat).astype(np.int32)
        self.threshold = np.concatenate(thr).astype(np.float64)
        self.left = np.concatenate(left).astype(np.int32)
        self.right = np.concatenate(right).astype(np.int32)
        self.prob = np.concatenate(prob).astype(np.float64)
        self.is_leaf = self.left == np.arange(len(self.left))
        self.roots = offsets.astype(np.int32)
        self.n_trees = len(trees)
        self.n_features = rf.n_features_in_
        self.depth = int(max(est.get_depth() for est in rf.estimators_))
        self.bias = float(self.prob[self.roots].mean())

    @property
    def n_nodes(self) -> int:
        return len(self.feature)

    def _path(self, x: np.ndarray) -> list[np.ndarray]:
        """Nœuds visités, niveau par niveau (tableaux de taille n_arbres). Le test « tous les
        arbres sont-ils arrivés à une feuille ? » n'est fait que tous les 8 niveaux : il coûte
        plus cher qu'un niveau de descente."""
        node = self.roots
        path = [node]
        for level in range(self.depth):
            node = np.where(x[self.feature[node]] <= self.threshold[node], self.left[node], self.right[node])
            path.append(node)
            if level % 8 == 7 and self.is_leaf[node].all():
                break
        return path

    def predict_proba_one(self, x: np.ndarray) -> float:
        return float(self.prob[self._path(np.asarray(x, dtype=np.float64))[-1]].mean())

    def explain_one(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """(probabilité, contributions par variable) ; probabilité = biais + somme."""
        path = np.stack(self._path(np.asarray(x, dtype=np.float64)))       # (niveaux, arbres)
        parent, child = path[:-1].ravel(), path[1:].ravel()
        moved = parent != child                                             # une feuille ne bouge plus
        contrib = np.bincount(self.feature[parent[moved]], minlength=self.n_features,
                              weights=self.prob[child[moved]] - self.prob[parent[moved]])
        return float(self.prob[path[-1]].mean()), contrib / self.n_trees
