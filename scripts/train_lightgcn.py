import os
import json
import time
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer


def get_output_dir(base_dir: str, model_name: str, dataset_name: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_dir, f"{model_name}-{dataset_name}-{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def find_latest_checkpoint(saved_dir: str, model_name: str, dataset_name: str) -> Optional[str]:
    if not os.path.isdir(saved_dir):
        return None
    candidates = []
    try:
        for fname in os.listdir(saved_dir):
            if not fname.endswith('.pth'):
                continue
            # only consider model checkpoints, not dataset/dataloader caches
            lower = fname.lower()
            if 'dataloader' in lower or 'dataset' in lower:
                continue
            name_ok = (model_name.lower() in lower) and (dataset_name.lower() in lower)
            if name_ok:
                full = os.path.join(saved_dir, fname)
                try:
                    ctime = os.path.getctime(full)
                except Exception:
                    ctime = 0.0
                candidates.append((ctime, full))
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _try_compute_final_embeddings(model: LightGCN) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Best-effort computation of final propagated embeddings.

    Tries multiple strategies depending on RecBole version. Returns CPU tensors or None if not possible.
    """
    model.eval()
    with torch.no_grad():
        # Strategy 1: some versions provide `computer()`
        if hasattr(model, "computer") and callable(getattr(model, "computer")):
            try:
                users_final, items_final = model.computer()
                return users_final.detach().cpu(), items_final.detach().cpu()
            except Exception:
                pass

        # Strategy 2: derive from get_ego_embeddings + normalized adjacency
        has_get_ego = hasattr(model, "get_ego_embeddings") and callable(getattr(model, "get_ego_embeddings"))
        norm_adj = None
        for attr in ("norm_adj_matrix", "Graph", "adj_t", "adj_mat"):
            if hasattr(model, attr):
                norm_adj = getattr(model, attr)
                break
        if has_get_ego and norm_adj is not None:
            try:
                users_emb, items_emb = model.get_ego_embeddings()
                embeddings = torch.cat([users_emb, items_emb], dim=0)
                all_embeddings = [embeddings]

                # Convert to a torch.sparse tensor if needed
                if not isinstance(norm_adj, torch.Tensor):
                    # Try to access underlying tensor-like data
                    try:
                        norm_adj = norm_adj.tocoo()
                        indices = torch.tensor([norm_adj.row, norm_adj.col], dtype=torch.long)
                        values = torch.tensor(norm_adj.data, dtype=embeddings.dtype, device=embeddings.device)
                        norm_adj = torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))
                        norm_adj = norm_adj.coalesce().to(embeddings.device)
                    except Exception:
                        norm_adj = None

                if isinstance(norm_adj, torch.Tensor) and norm_adj.is_sparse:
                    for _ in range(int(getattr(model, "n_layers", 1))):
                        embeddings = torch.sparse.mm(norm_adj, embeddings)
                        all_embeddings.append(embeddings)
                    all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

                    n_users = getattr(model, "n_users", model.user_embedding.weight.shape[0])
                    n_items = getattr(model, "n_items", model.item_embedding.weight.shape[0])
                    users_final, items_final = torch.split(all_embeddings, [n_users, n_items], dim=0)
                    return users_final.detach().cpu(), items_final.detach().cpu()
            except Exception:
                pass

    return None


def compute_embeddings(model: LightGCN) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return initial and final (propagated when possible) user/item embeddings on CPU tensors."""
    model.eval()
    with torch.no_grad():
        initial_user = model.user_embedding.weight.detach().cpu()
        initial_item = model.item_embedding.weight.detach().cpu()

        final_pair = _try_compute_final_embeddings(model)
        if final_pair is None:
            # Fallback: if we cannot compute propagated embeddings, reuse initial
            final_user, final_item = initial_user.clone(), initial_item.clone()
        else:
            final_user, final_item = final_pair
    return initial_user, initial_item, final_user, final_item


def save_embeddings(out_dir: str,
                    initial_user: torch.Tensor,
                    initial_item: torch.Tensor,
                    final_user: torch.Tensor,
                    final_item: torch.Tensor) -> None:
    torch.save(initial_user, os.path.join(out_dir, "user_embeddings_initial.pt"))
    torch.save(initial_item, os.path.join(out_dir, "item_embeddings_initial.pt"))
    torch.save(final_user, os.path.join(out_dir, "user_embeddings_final.pt"))
    torch.save(final_item, os.path.join(out_dir, "item_embeddings_final.pt"))

    # Also save as .npy for convenience
    np.save(os.path.join(out_dir, "user_embeddings_initial.npy"), initial_user.numpy())
    np.save(os.path.join(out_dir, "item_embeddings_initial.npy"), initial_item.numpy())
    np.save(os.path.join(out_dir, "user_embeddings_final.npy"), final_user.numpy())
    np.save(os.path.join(out_dir, "item_embeddings_final.npy"), final_item.numpy())


def save_id_mappings(out_dir: str, dataset) -> None:
    # dataset.uid_field (e.g., 'user_id') and dataset.iid_field (e.g., 'item_id')
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    num_users = int(dataset.num(uid_field))
    num_items = int(dataset.num(iid_field))

    user_inner_ids = np.arange(num_users)
    item_inner_ids = np.arange(num_items)

    # Convert inner ids to original tokens
    user_tokens = dataset.id2token(uid_field, user_inner_ids)
    item_tokens = dataset.id2token(iid_field, item_inner_ids)

    # Save as JSON for readability
    with open(os.path.join(out_dir, "user_id_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"inner_id": user_inner_ids.tolist(), "token": list(map(str, user_tokens))}, f)
    with open(os.path.join(out_dir, "item_id_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"inner_id": item_inner_ids.tolist(), "token": list(map(str, item_tokens))}, f)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LightGCN with different test data splitting approaches')
    parser.add_argument('--test', type=str, choices=['LOO', 'ratio'], default='LOO',
                       help='Test data splitting approach: LOO (leave-one-out) or ratio (8:1:1)')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Minimal but solid config for LightGCN on ml-100k
    config_dict = {
        "model": "LightGCN",
        "dataset": "ml-100k",
        # training/eval
        "epochs": 300,
        "train_batch_size": 2048,
        "eval_batch_size": 4096,
        "learning_rate": 1e-3,
        # New-style negative sampling args (replace deprecated 'neg_sampling')
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "order": "TO",
            "mode": "full",
            "group_by": "user",
        },
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
        "topk": [10],
        "valid_metric": "MRR@10",
        "valid_metric_bigger": True,
        "metric_decimal_place": 4,
        # model hyperparams
        "embedding_size": 64,
        "n_layers": 2,
        "reg_weight": 1e-5,
        # runtime
        "device": device,
        # speed up resume
        "save_dataset": True,
        "save_dataloaders": True,
        # make results reproducible (seed is respected by RecBole)
        "seed": 42,
    }
    
    # Configure data splitting based on user choice
    if args.test == 'ratio':
        # Use ratio split (8:1:1 for train:valid:test)
        config_dict["split_ratio"] = [0.8, 0.1, 0.1]
        # Remove the LS split configuration for ratio approach
        if "eval_args" in config_dict and "split" in config_dict["eval_args"]:
            del config_dict["eval_args"]["split"]
        print("Using ratio split approach (8:1:1 for train:valid:test)")
    else:
        # Use LOO approach (default RecBole behavior)
        print("Using leave-one-out (LOO) approach for test data")

    # Build RecBole objects with preflight checks
    try:
        config = Config(config_dict=config_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to build RecBole Config: {e}")

    try:
        dataset = create_dataset(config)
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset {config['dataset']}: {e}")

    try:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    except Exception as e:
        raise RuntimeError(f"Data preparation failed: {e}")

    # Initialize model and trainer
    try:
        model = LightGCN(config, train_data.dataset).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LightGCN: {e}")

    # Resume from latest checkpoint if available (load weights only)
    try:
        latest_ckpt = find_latest_checkpoint(saved_dir="saved", model_name="LightGCN", dataset_name=str(config["dataset"]))
        if latest_ckpt is not None:
            print(f"Resuming from checkpoint weights: {latest_ckpt}")
            state = torch.load(latest_ckpt, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                # Direct state_dict
                model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"Warning: failed to load resume checkpoint due to: {e}")

    try:
        # Preflight: ensure we can compute embeddings before long training
        _ = compute_embeddings(model)
    except Exception as e:
        raise RuntimeError(f"Preflight embedding computation failed (check RecBole version compatibility): {e}")

    try:
        trainer = Trainer(config, model)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Trainer: {e}")

    print(f"Starting training with {args.test} test data approach...")

    # Train (and save best checkpoint)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=True,
    )

    # Load best checkpoint explicitly (PyTorch 2.6 defaults weights_only=True causing RecBole torch.load to fail)
    if os.path.isfile(trainer.saved_model_file):
        try:
            state = torch.load(trainer.saved_model_file, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Warning: failed to load saved checkpoint with weights_only=False due to: {e}. Using in-memory model.")
    else:
        print(f"Warning: saved model file not found at {trainer.saved_model_file}. Using in-memory model.")

    # Evaluate on test set using the in-memory model (avoid internal torch.load)
    try:
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    except Exception as e:
        print(f"Warning: test evaluation failed: {e}")
        test_result = {"error": str(e)}

    # Compute and save embeddings + id mappings + metadata
    initial_user, initial_item, final_user, final_item = compute_embeddings(model)

    out_dir = get_output_dir(base_dir=os.path.join("run", "LightGCN"),
                             model_name="LightGCN",
                             dataset_name=f"{config['dataset']}-{args.test}")

    save_embeddings(out_dir, initial_user, initial_item, final_user, final_item)
    save_id_mappings(out_dir, dataset)

    # Save run metadata (robust to RecBole/PyTorch versions)
    def _sanitize(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        return str(obj)

    try:
        if hasattr(config, "final_config_dict"):
            cfg_items = getattr(config, "final_config_dict").items()
        elif hasattr(config, "config_dict"):
            cfg_items = getattr(config, "config_dict").items()
        else:
            cfg_items = []
        cfg_dict = {str(k): _sanitize(v) for k, v in cfg_items}
    except Exception:
        cfg_dict = {}

    meta = {
        "model": "LightGCN",
        "dataset": str(config["dataset"]),
        "test_approach": args.test,
        "config": cfg_dict,
        "best_valid_score": float(best_valid_score) if best_valid_score is not None else None,
        "best_valid_result": _sanitize(best_valid_result),
        "test_result": _sanitize(test_result),
        "saved_model_file": trainer.saved_model_file,
        "device": device,
    }
    try:
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save metadata.json due to: {e}")

    print(f"Saved embeddings and metadata to: {out_dir}")


if __name__ == "__main__":
    main()
    
    # Usage examples:
    # python train_lightgcn.py --test LOO      # Use leave-one-out approach (default)
    # python train_lightgcn.py --test ratio    # Use ratio split (8:1:1)


