"""Train preliminary full-catalog H&M recommendation baselines.

The prepared sample is created by prepare.py. Model selection uses validation
NDCG@10 only. The test set is evaluated once after selecting the best epoch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


KS = (10, 20)


def labels_by_user(pairs: np.ndarray) -> dict[int, set[int]]:
    labels: dict[int, set[int]] = defaultdict(set)
    for user, item in pairs:
        labels[int(user)].add(int(item))
    return dict(labels)


def summarize_rankings(
    labels: dict[int, set[int]], ranked_items: dict[int, np.ndarray]
) -> dict[str, float | int]:
    totals = {"Recall@10": 0.0, "Recall@20": 0.0, "NDCG@10": 0.0,
              "NDCG@20": 0.0, "MAP@12": 0.0}
    discounts = [1.0 / math.log2(position + 2) for position in range(20)]
    for user, relevant in labels.items():
        ranked = ranked_items[user]
        hit_positions = [position for position, item in enumerate(ranked[:20]) if int(item) in relevant]
        for k in KS:
            hits = sum(position < k for position in hit_positions)
            totals[f"Recall@{k}"] += hits / len(relevant)
            dcg = sum(discounts[position] for position in hit_positions if position < k)
            ideal = sum(discounts[: min(len(relevant), k)])
            totals[f"NDCG@{k}"] += dcg / ideal
        ap = 0.0
        hits_so_far = 0
        for position in hit_positions:
            if position >= 12:
                break
            hits_so_far += 1
            ap += hits_so_far / (position + 1)
        totals["MAP@12"] += ap / min(len(relevant), 12)
    count = len(labels)
    return {"users": count, **{key: value / count for key, value in totals.items()}}


@torch.no_grad()
def rank_embeddings(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    users: list[int],
    batch_users: int,
) -> dict[int, np.ndarray]:
    ranking = {}
    for start in range(0, len(users), batch_users):
        batch = users[start : start + batch_users]
        scores = user_vectors[batch] @ item_vectors.T
        indices = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
        ranking.update({user: indices[row] for row, user in enumerate(batch)})
    return ranking


def evaluate_embeddings(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    labels: dict[int, set[int]],
    batch_users: int = 64,
) -> dict[str, float | int]:
    ranked = rank_embeddings(user_vectors, item_vectors, sorted(labels), batch_users)
    return summarize_rankings(labels, ranked)


def evaluate_popularity(
    train_pairs: np.ndarray,
    n_items: int,
    labels: dict[int, set[int]],
) -> dict[str, float | int]:
    popularity = np.bincount(train_pairs[:, 1], minlength=n_items)
    ranked = np.lexsort((np.arange(n_items), -popularity))[:20]
    return summarize_rankings(labels, {user: ranked for user in labels})


def draw_negatives(
    users: np.ndarray,
    n_items: int,
    positive_keys: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, n_items, size=len(users), dtype=np.int32)
    while True:
        keys = users.astype(np.int64) * n_items + negatives
        locations = np.searchsorted(positive_keys, keys)
        locations = np.minimum(locations, len(positive_keys) - 1)
        collision = positive_keys[locations] == keys
        if not collision.any():
            return negatives
        negatives[collision] = rng.integers(
            0, n_items, size=int(collision.sum()), dtype=np.int32
        )


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dimension: int):
        super().__init__()
        self.users = nn.Embedding(n_users, dimension)
        self.items = nn.Embedding(n_items, dimension)
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)

    def vectors(self):
        return self.users.weight, self.items.weight


class LightGCN(BPRMF):
    def __init__(
        self, n_users: int, n_items: int, dimension: int, train_pairs: np.ndarray, device
    ):
        super().__init__(n_users, n_items, dimension)
        self.n_users = n_users
        self.adjacency = self.make_adjacency(train_pairs, n_users, n_items, device)

    @staticmethod
    def make_adjacency(train_pairs, n_users, n_items, device):
        left = train_pairs[:, 0].astype(np.int64)
        right = train_pairs[:, 1].astype(np.int64) + n_users
        row = np.concatenate((left, right))
        col = np.concatenate((right, left))
        degrees = np.bincount(row, minlength=n_users + n_items)
        weights = 1.0 / np.sqrt(degrees[row] * degrees[col])
        indices = torch.from_numpy(np.stack((row, col))).to(device)
        values = torch.from_numpy(weights.astype(np.float32)).to(device)
        return torch.sparse_coo_tensor(
            indices, values, size=(n_users + n_items, n_users + n_items), device=device
        ).coalesce()

    def vectors(self):
        initial = torch.cat((self.users.weight, self.items.weight), dim=0)
        one_hop = torch.sparse.mm(self.adjacency, initial)
        two_hop = torch.sparse.mm(self.adjacency, one_hop)
        combined = (initial + one_hop + two_hop) / 3.0
        return combined[: self.n_users], combined[self.n_users :]


@torch.no_grad()
def validate_model(model, labels, batch_users):
    model.eval()
    users, items = model.vectors()
    return evaluate_embeddings(users, items, labels, batch_users)


def train_model(
    name: str,
    train_pairs: np.ndarray,
    n_users: int,
    n_items: int,
    validation_labels: dict[int, set[int]],
    test_labels: dict[int, set[int]],
    args,
    device,
) -> dict:
    if name == "BPR-MF":
        model = BPRMF(n_users, n_items, args.dimension).to(device)
        batch_size = args.mf_batch_size
        steps_per_epoch = math.ceil(len(train_pairs) / batch_size)
    else:
        model = LightGCN(n_users, n_items, args.dimension, train_pairs, device).to(device)
        batch_size = args.gcn_batch_size
        steps_per_epoch = args.gcn_steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + (0 if name == "BPR-MF" else 1))
    positive_keys = np.sort(
        train_pairs[:, 0].astype(np.int64) * n_items + train_pairs[:, 1]
    )
    best_state = None
    best_validation = -1.0
    best_epoch = 0
    epochs = args.mf_epochs if name == "BPR-MF" else args.gcn_epochs
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_pairs)) if name == "BPR-MF" else None
        total_loss = 0.0
        for step in range(steps_per_epoch):
            if order is not None:
                indices = order[step * batch_size : (step + 1) * batch_size]
            else:
                indices = rng.integers(0, len(train_pairs), size=batch_size)
            batch = train_pairs[indices]
            user_np = batch[:, 0]
            pos_np = batch[:, 1]
            neg_np = draw_negatives(user_np, n_items, positive_keys, rng)
            user = torch.as_tensor(user_np, dtype=torch.long, device=device)
            positive = torch.as_tensor(pos_np, dtype=torch.long, device=device)
            negative = torch.as_tensor(neg_np, dtype=torch.long, device=device)

            user_vectors, item_vectors = model.vectors()
            user_vec = user_vectors[user]
            positive_vec = item_vectors[positive]
            negative_vec = item_vectors[negative]
            margin = (user_vec * (positive_vec - negative_vec)).sum(dim=1)
            loss = F.softplus(-margin).mean()
            reg = (
                model.users(user).square().sum(dim=1)
                + model.items(positive).square().sum(dim=1)
                + model.items(negative).square().sum(dim=1)
            ).mean() / (2.0 * args.dimension)
            loss = loss + args.regularization * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        validation = validate_model(model, validation_labels, args.eval_batch_users)
        row = {
            "epoch": epoch,
            "loss": total_loss / steps_per_epoch,
            "validation_NDCG@10": validation["NDCG@10"],
            "validation_Recall@10": validation["Recall@10"],
        }
        history.append(row)
        print(json.dumps({"model": name, **row}), flush=True)
        if validation["NDCG@10"] > best_validation:
            best_validation = validation["NDCG@10"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    validation = validate_model(model, validation_labels, args.eval_batch_users)
    test = validate_model(model, test_labels, args.eval_batch_users)
    return {
        "model": name,
        "best_epoch": best_epoch,
        "selection_metric": "validation NDCG@10",
        "validation": validation,
        "test": test,
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "config": {
            "dimension": args.dimension,
            "epochs": epochs,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": args.learning_rate,
            "regularization": args.regularization,
            "seed": args.seed,
            "device": str(device),
            "negative_sampling": "Uniform over candidate items, rejecting training positives",
            "gcn_layers": 2 if name == "LightGCN" else 0,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=root / "results" / "hm" / "sample_50k_seed20260923",
    )
    parser.add_argument("--models", nargs="+", default=["MostPop", "BPR-MF", "LightGCN"])
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--mf-epochs", type=int, default=8)
    parser.add_argument("--gcn-epochs", type=int, default=8)
    parser.add_argument("--mf-batch-size", type=int, default=16384)
    parser.add_argument("--gcn-batch-size", type=int, default=65536)
    parser.add_argument("--gcn-steps-per-epoch", type=int, default=22)
    parser.add_argument("--eval-batch-users", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not set(args.models) <= {"MostPop", "BPR-MF", "LightGCN"}:
        raise ValueError("Unknown model requested")
    output = args.output or args.sample_dir / "baseline_results.json"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.sample_dir / "interactions.npz")
    train_pairs = data["train_pairs"]
    validation_labels = labels_by_user(data["validation_pairs"])
    test_labels = labels_by_user(data["test_pairs"])
    n_users = len(data["user_ids"])
    n_items = len(data["item_ids"])
    result = {
        "sample_manifest": str(args.sample_dir / "manifest.json"),
        "n_train_pairs": len(train_pairs),
        "n_users": n_users,
        "n_candidate_items": n_items,
        "evaluation": "Full ranking over candidate items; prior purchases are not masked",
        "models": {},
    }
    if "MostPop" in args.models:
        result["models"]["MostPop"] = {
            "model": "MostPop",
            "validation": evaluate_popularity(train_pairs, n_items, validation_labels),
            "test": evaluate_popularity(train_pairs, n_items, test_labels),
        }
        print(json.dumps(result["models"]["MostPop"]), flush=True)

    for name in ("BPR-MF", "LightGCN"):
        if name in args.models:
            result["models"][name] = train_model(
                name,
                train_pairs,
                n_users,
                n_items,
                validation_labels,
                test_labels,
                args,
                device,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

