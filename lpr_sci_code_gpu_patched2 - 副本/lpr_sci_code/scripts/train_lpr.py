#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lpr.common import ensure_dir, get_device, load_config, read_jsonl, set_seed, write_jsonl
from lpr.data import load_standard_dataset
from lpr.models import KnowledgeBackgroundEncoder, KnowledgeTracer, LPRModel, PolicyValueNet, RGCNEncoder, TimeAwarePreferenceEncoder, TransEEncoder
from lpr.rl import RewardWeights
from lpr.trainers import LPRTrainer, make_task_loaders, train_transe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the learning path recommendation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--kt_ckpt", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    dataset_dir = args.dataset_dir or cfg.data.dataset_dir
    output_dir = ensure_dir(args.output_dir or cfg.output.output_dir)
    device = get_device(getattr(cfg.train, "device", None))

    dataset = load_standard_dataset(dataset_dir)
    train_loader, val_loader, test_loader = make_task_loaders(
        dataset,
        batch_size=cfg.train.lpr_batch_size,
        device=device,
        num_workers=getattr(cfg.train, "num_workers", 0),
        pin_memory=getattr(cfg.train, "pin_memory", None),
    )
    print(f"[INFO] Training LPR on device={device}. cuda_available={torch.cuda.is_available()}")

    kt_ckpt = args.kt_ckpt or getattr(cfg.output, "kt_ckpt", None) or str(output_dir / "kt_best.pt")
    if not Path(kt_ckpt).exists():
        raise FileNotFoundError(f"KT checkpoint not found: {kt_ckpt}. Please run scripts/train_kt.py first.")
    num_nodes = int(dataset.graph.num_nodes)
    kt_model = KnowledgeTracer(num_nodes=num_nodes, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
    kt_model.load_state_dict(torch.load(kt_ckpt, map_location=device))
    kt_model.to(device).eval()

    graph_type = cfg.variant.graph_type
    if graph_type.lower() == "rgcn":
        graph_encoder = RGCNEncoder(
            num_nodes=num_nodes,
            num_relations=dataset.graph.num_relations,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_gnn_layers,
            dropout=cfg.model.dropout,
        )
    elif graph_type.lower() == "transe":
        graph_encoder = TransEEncoder(
            num_nodes=num_nodes,
            num_relations=dataset.graph.num_relations,
            hidden_dim=cfg.model.hidden_dim,
        )
        graph_ckpt = output_dir / "graph_best.pt"
        report = train_transe(
            graph_encoder,
            dataset.graph,
            device=device,
            epochs=cfg.train.graph_epochs,
            lr=cfg.train.graph_lr,
            margin=cfg.train.graph_margin,
            ckpt_path=str(graph_ckpt),
        )
        graph_encoder.load_state_dict(torch.load(graph_ckpt, map_location=device))
        with open(output_dir / "graph_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"best": report.best_metric, "history": report.history}, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    preference_encoder = TimeAwarePreferenceEncoder(hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout)
    kb_encoder = KnowledgeBackgroundEncoder(hidden_dim=cfg.model.hidden_dim, mode=cfg.variant.kb_mode, dropout=cfg.model.dropout)
    policy = PolicyValueNet(hidden_dim=cfg.model.hidden_dim, num_nodes=num_nodes, dropout=cfg.model.dropout)
    model = LPRModel(graph_encoder, preference_encoder, kb_encoder, policy, hidden_dim=cfg.model.hidden_dim)

    trainer = LPRTrainer(
        model=model,
        graph=dataset.graph,
        kt_model=kt_model,
        device=device,
        reward_mode=cfg.variant.reward_mode,
        reward_weights=RewardWeights(**cfg.reward.to_dict()),
        candidate_mode=cfg.variant.candidate_mode,
        path_len=cfg.data.path_len,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
    )

    bc_ckpt = output_dir / "lpr_bc_best.pt"
    rl_ckpt = output_dir / "lpr_rl_best.pt"
    all_reports = {}
    if cfg.train.bc_epochs > 0:
        bc_report = trainer.behavior_clone(
            train_loader,
            val_loader,
            epochs=cfg.train.bc_epochs,
            lr=cfg.train.bc_lr,
            ckpt_path=str(bc_ckpt),
        )
        if bc_ckpt.exists():
            model.load_state_dict(torch.load(bc_ckpt, map_location=device))
        all_reports["behavior_cloning"] = {"best": bc_report.best_metric, "history": bc_report.history}
    if cfg.train.rl_epochs > 0:
        rl_report = trainer.rl_finetune(
            train_loader,
            val_loader,
            epochs=cfg.train.rl_epochs,
            lr=cfg.train.rl_lr,
            ppo_epochs=cfg.train.ppo_epochs,
            clip_eps=cfg.train.clip_eps,
            value_coef=cfg.train.value_coef,
            entropy_coef=cfg.train.entropy_coef,
            ckpt_path=str(rl_ckpt),
        )
        if rl_ckpt.exists():
            model.load_state_dict(torch.load(rl_ckpt, map_location=device))
        all_reports["rl"] = {"best": rl_report.best_metric, "history": rl_report.history}

    val_metrics = trainer.evaluate(val_loader, greedy=True)
    test_metrics = trainer.evaluate(test_loader, greedy=True)
    preds = trainer.generate_paths(test_loader, greedy=True)
    write_jsonl(preds[: min(len(preds), 200)], output_dir / "test_predictions_sample.jsonl")

    result = {
        "variant": cfg.variant.to_dict(),
        "validation": val_metrics,
        "test": test_metrics,
        "reports": all_reports,
    }
    with open(output_dir / "lpr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # Save final checkpoint for later evaluation.
    torch.save(model.state_dict(), output_dir / "lpr_final.pt")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
