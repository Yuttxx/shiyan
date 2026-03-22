from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baselines import GRU4RecBaseline
from .common import ensure_dir, pad_float_sequences, pad_sequences, sequence_mask
from .data import GraphStore, StandardDataset, collate_kt, collate_tasks
from .metrics import (
    aggregate,
    difficulty_smoothness,
    hit_rate,
    mastery_gain_metric,
    mrr,
    ndcg_at_k,
    prerequisite_violation_rate,
    review_coverage,
)
from .models import KnowledgeTracer, LPRModel, TransEEncoder
from .rl import CandidateGenerator, RewardWeights, append_step, compute_gae, dense_reward, sparse_reward


@dataclass
class TrainReport:
    best_metric: float
    history: List[Dict[str, float]]


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def dataloader_kwargs(device: torch.device, num_workers: int = 0, pin_memory: Optional[bool] = None) -> Dict[str, Any]:
    use_pin = device.type == "cuda" if pin_memory is None else bool(pin_memory and device.type == "cuda")
    kwargs: Dict[str, Any] = {"num_workers": int(num_workers), "pin_memory": use_pin}
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
    return kwargs



def history_mastery_from_feedback(history: Sequence[int], correct: Sequence[float]) -> Dict[int, float]:
    total: Dict[int, float] = {}
    count: Dict[int, float] = {}
    for cid, c in zip(history, correct):
        total[int(cid)] = total.get(int(cid), 0.0) + float(max(c, 0.0))
        count[int(cid)] = count.get(int(cid), 0.0) + 1.0
    return {cid: total[cid] / max(count[cid], 1.0) for cid in total}



def lists_to_batch(
    histories: List[List[int]],
    corrects: List[List[float]],
    deltas: List[List[float]],
    targets: List[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([len(h) for h in histories], dtype=torch.long, device=device)
    return {
        "history": pad_sequences(histories, pad_value=0).to(device),
        "history_correct": pad_float_sequences(corrects, pad_value=-1.0).to(device),
        "history_deltas": pad_float_sequences(deltas, pad_value=0.0).to(device),
        "lengths": lengths,
        "target": torch.tensor(targets, dtype=torch.long, device=device),
    }



def evaluate_kt(model: KnowledgeTracer, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    y_true: List[float] = []
    y_prob: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with autocast(enabled=device.type == "cuda"):
                logits = model(batch["concepts"], batch["correct"], batch["deltas"], batch["lengths"])
                mask = sequence_mask(batch["lengths"], max_len=batch["concepts"].size(1))
                target_logits = logits.gather(2, batch["target_concepts"].unsqueeze(-1)).squeeze(-1)
            valid = mask & (batch["target_correct"] >= 0)
            loss = F.binary_cross_entropy_with_logits(target_logits[valid], batch["target_correct"][valid])
            losses.append(float(loss.item()))
            y_true.extend(batch["target_correct"][valid].detach().cpu().tolist())
            y_prob.extend(torch.sigmoid(target_logits[valid]).detach().cpu().tolist())
    auc = 0.5
    if len(set(np.array(y_true).astype(int).tolist())) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    return {"loss": float(np.mean(losses)), "auc": auc}



def train_knowledge_tracer(
    model: KnowledgeTracer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    ckpt_path: Optional[str] = None,
) -> TrainReport:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=device.type == "cuda")
    best_auc = -1.0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"KT epoch {epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(batch["concepts"], batch["correct"], batch["deltas"], batch["lengths"])
                mask = sequence_mask(batch["lengths"], max_len=batch["concepts"].size(1))
                target_logits = logits.gather(2, batch["target_concepts"].unsqueeze(-1)).squeeze(-1)
                valid = mask & (batch["target_correct"] >= 0)
                loss = F.binary_cross_entropy_with_logits(target_logits[valid], batch["target_correct"][valid])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_metric = evaluate_kt(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metric}
        history.append(record)
        if val_metric["auc"] > best_auc:
            best_auc = val_metric["auc"]
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
    return TrainReport(best_metric=best_auc, history=history)



def train_transe(
    model: TransEEncoder,
    graph: GraphStore,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    margin: float = 1.0,
    ckpt_path: Optional[str] = None,
) -> TrainReport:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    relation2id = {rel: idx for idx, rel in enumerate(graph.relations)}
    triples: List[Tuple[int, int, int]] = []
    for rel, edges in graph.edges_by_rel.items():
        r = relation2id[rel]
        triples.extend([(s, r, d) for s, d, _ in edges])
    triples = list({(s, r, d) for s, r, d in triples})
    history: List[Dict[str, float]] = []
    best = 1e9
    for epoch in range(1, epochs + 1):
        np.random.shuffle(triples)
        losses = []
        for s, r, d in triples:
            h = torch.tensor([s], dtype=torch.long, device=device)
            rel = torch.tensor([r], dtype=torch.long, device=device)
            t = torch.tensor([d], dtype=torch.long, device=device)
            neg_d = torch.tensor([np.random.randint(0, graph.num_nodes)], dtype=torch.long, device=device)
            pos = model.score(h, rel, t)
            neg = model.score(h, rel, neg_d)
            loss = F.relu(margin - pos + neg).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        avg_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": avg_loss})
        if avg_loss < best:
            best = avg_loss
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
    return TrainReport(best_metric=-best, history=history)



def train_gru4rec(
    model: GRU4RecBaseline,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    ckpt_path: Optional[str] = None,
) -> TrainReport:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = 1e9
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch["history"], batch["correct"], batch["deltas"], batch["lengths"])
            loss = F.cross_entropy(logits, batch["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                with autocast(enabled=device.type == "cuda"):
                    logits = model(batch["history"], batch["correct"], batch["deltas"], batch["lengths"])
                    val_losses.append(float(F.cross_entropy(logits, batch["label"]).item()))
        avg_val = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": avg_val})
        if avg_val < best:
            best = avg_val
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
    return TrainReport(best_metric=-best, history=history)


class LPRTrainer:
    def __init__(
        self,
        model: LPRModel,
        graph: GraphStore,
        kt_model: KnowledgeTracer,
        device: torch.device,
        reward_mode: str = "dense",
        reward_weights: Optional[RewardWeights] = None,
        candidate_mode: str = "review_augmented",
        path_len: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.model = model.to(device)
        self.graph = graph
        self.kt_model = kt_model.to(device)
        self.kt_model.eval()
        self.device = device
        self.reward_mode = reward_mode
        self.reward_weights = reward_weights or RewardWeights()
        self.candidate_generator = CandidateGenerator(graph, mode=candidate_mode)
        self.path_len = path_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _candidate_mask(self, histories: List[List[int]], corrects: List[List[float]], targets: List[int]) -> torch.Tensor:
        mastery_list = [history_mastery_from_feedback(h, c) for h, c in zip(histories, corrects)]
        return self.candidate_generator.mask_from_histories(histories, targets, mastery_list).to(self.device)

    def _step_reward(
        self,
        history: List[int],
        correct: List[float],
        deltas: List[float],
        action: int,
        target: int,
        path_done: bool,
    ) -> Tuple[float, float]:
        mastery_dict = history_mastery_from_feedback(history, correct)
        if self.reward_mode == "dense":
            return dense_reward(
                self.graph,
                self.kt_model,
                history,
                correct,
                deltas,
                action,
                target,
                mastery_dict,
                self.reward_weights,
                self.device,
                path_done,
            )
        reward = sparse_reward(self.graph, history, action, target, mastery_dict, path_done)
        next_correct = 1.0 if reward > 0 else 0.0
        return reward, next_correct

    def behavior_clone(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        lr: float = 1e-3,
        ckpt_path: Optional[str] = None,
    ) -> TrainReport:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.device.type == "cuda")
        best = 1e9
        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f"BC epoch {epoch}", leave=False):
                batch = move_batch_to_device(batch, self.device)
                histories = [row[:l].tolist() for row, l in zip(batch["history"], batch["lengths"])]
                corrects = [row[:l].tolist() for row, l in zip(batch["history_correct"], batch["lengths"])]
                targets = batch["target"].tolist()
                action_mask = self._candidate_mask(histories, corrects, targets)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.device.type == "cuda"):
                    logits, _, _ = self.model(
                        self.graph,
                        batch["history"],
                        batch["history_correct"],
                        batch["history_deltas"],
                        batch["lengths"],
                        batch["target"],
                    )
                    labels = batch["future"][:, 0]
                    logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
            val_loss = self._behavior_clone_eval(val_loader)
            record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
            history.append(record)
            if val_loss < best:
                best = val_loss
                if ckpt_path:
                    torch.save(self.model.state_dict(), ckpt_path)
        return TrainReport(best_metric=-best, history=history)

    @torch.no_grad()
    def _behavior_clone_eval(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            histories = [row[:l].tolist() for row, l in zip(batch["history"], batch["lengths"])]
            corrects = [row[:l].tolist() for row, l in zip(batch["history_correct"], batch["lengths"])]
            targets = batch["target"].tolist()
            action_mask = self._candidate_mask(histories, corrects, targets)
            with autocast(enabled=self.device.type == "cuda"):
                logits, _, _ = self.model(
                    self.graph,
                    batch["history"],
                    batch["history_correct"],
                    batch["history_deltas"],
                    batch["lengths"],
                    batch["target"],
                )
                logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
                labels = batch["future"][:, 0]
                losses.append(float(F.cross_entropy(logits, labels).item()))
        return float(np.mean(losses))

    def rl_finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        ppo_epochs: int = 2,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ckpt_path: Optional[str] = None,
    ) -> TrainReport:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.device.type == "cuda")
        best = -1e9
        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            batch_records = []
            for batch in tqdm(train_loader, desc=f"RL epoch {epoch}", leave=False):
                batch = move_batch_to_device(batch, self.device)
                rollout = self._collect_rollout(batch)
                if not rollout:
                    continue
                loss_epoch = 0.0
                for _ in range(ppo_epochs):
                    total_loss = 0.0
                    n_steps = 0
                    for step in rollout:
                        with autocast(enabled=self.device.type == "cuda"):
                            logits, value, _ = self.model(
                                self.graph,
                                step["history"],
                                step["history_correct"],
                                step["history_deltas"],
                                step["lengths"],
                                step["target"],
                            )
                            logits = logits.masked_fill(~step["action_mask"], torch.finfo(logits.dtype).min)
                            probs = torch.softmax(logits, dim=-1)
                            dist = torch.distributions.Categorical(probs=probs)
                            new_log_prob = dist.log_prob(step["action"])
                            ratio = torch.exp(new_log_prob - step["old_log_prob"])
                            unclipped = ratio * step["advantage"]
                            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * step["advantage"]
                            policy_loss = -torch.min(unclipped, clipped).mean()
                            value_loss = F.mse_loss(value, step["return"])
                            entropy = dist.entropy().mean()
                            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                            total_loss = total_loss + loss
                        n_steps += 1
                    total_loss = total_loss / max(n_steps, 1)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    loss_epoch += float(total_loss.item())
                batch_records.append(loss_epoch / max(ppo_epochs, 1))
            val_metrics = self.evaluate(val_loader, greedy=True)
            record = {"epoch": epoch, "train_loss": float(np.mean(batch_records) if batch_records else 0.0), **val_metrics}
            history.append(record)
            target_metric = val_metrics.get("mastery_gain", 0.0) + val_metrics.get("hit_rate", 0.0)
            if target_metric > best:
                best = target_metric
                if ckpt_path:
                    torch.save(self.model.state_dict(), ckpt_path)
        return TrainReport(best_metric=best, history=history)

    def _collect_rollout(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        histories = [row[:l].tolist() for row, l in zip(batch["history"], batch["lengths"])]
        corrects = [row[:l].tolist() for row, l in zip(batch["history_correct"], batch["lengths"])]
        deltas = [row[:l].tolist() for row, l in zip(batch["history_deltas"], batch["lengths"])]
        targets = batch["target"].tolist()
        per_sample_rewards: List[List[float]] = [[] for _ in histories]
        per_sample_values: List[List[float]] = [[] for _ in histories]
        per_sample_steps: List[List[Dict[str, torch.Tensor]]] = [[] for _ in histories]
        for step_idx in range(self.path_len):
            state_batch = lists_to_batch(histories, corrects, deltas, targets, self.device)
            action_mask = self._candidate_mask(histories, corrects, targets)
            act_out = self.model.act(
                self.graph,
                state_batch["history"],
                state_batch["history_correct"],
                state_batch["history_deltas"],
                state_batch["lengths"],
                state_batch["target"],
                action_mask,
                greedy=False,
            )
            actions = act_out["action"].detach().cpu().tolist()
            rewards = []
            next_corrects = []
            for i, (history, corr, delta, action, target) in enumerate(zip(histories, corrects, deltas, actions, targets)):
                reward, next_correct = self._step_reward(
                    history,
                    corr,
                    delta,
                    action,
                    target,
                    path_done=(step_idx == self.path_len - 1),
                )
                rewards.append(reward)
                next_corrects.append(next_correct)
                per_sample_rewards[i].append(float(reward))
                per_sample_values[i].append(float(act_out["value"][i].item()))
                per_sample_steps[i].append(
                    {
                        "history": state_batch["history"][i].detach().clone(),
                        "history_correct": state_batch["history_correct"][i].detach().clone(),
                        "history_deltas": state_batch["history_deltas"][i].detach().clone(),
                        "length": int(state_batch["lengths"][i].item()),
                        "target": int(state_batch["target"][i].item()),
                        "action_mask": action_mask[i].detach().clone(),
                        "action": int(actions[i]),
                        "old_log_prob": float(act_out["log_prob"][i].item()),
                    }
                )
            histories = [h + [a] for h, a in zip(histories, actions)]
            corrects = [c + [nc] for c, nc in zip(corrects, next_corrects)]
            deltas = [d + [d[-1] if d else 0.0] for d in deltas]
        flat_steps: List[Dict[str, torch.Tensor]] = []
        for sample_steps, rewards, values in zip(per_sample_steps, per_sample_rewards, per_sample_values):
            returns, advantages = compute_gae(rewards, values, gamma=self.gamma, lam=self.gae_lambda)
            for step, ret, adv in zip(sample_steps, returns, advantages):
                flat_steps.append(
                    {
                        "history": step["history"].unsqueeze(0).to(self.device),
                        "history_correct": step["history_correct"].unsqueeze(0).to(self.device),
                        "history_deltas": step["history_deltas"].unsqueeze(0).to(self.device),
                        "lengths": torch.tensor([step["length"]], dtype=torch.long, device=self.device),
                        "target": torch.tensor([step["target"]], dtype=torch.long, device=self.device),
                        "action_mask": step["action_mask"].unsqueeze(0).to(self.device),
                        "action": torch.tensor([step["action"]], dtype=torch.long, device=self.device),
                        "old_log_prob": torch.tensor([step["old_log_prob"]], dtype=torch.float32, device=self.device),
                        "return": torch.tensor([ret], dtype=torch.float32, device=self.device),
                        "advantage": torch.tensor([adv], dtype=torch.float32, device=self.device),
                    }
                )
        if flat_steps:
            adv = torch.cat([s["advantage"] for s in flat_steps], dim=0)
            adv = (adv - adv.mean()) / adv.std().clamp(min=1e-6)
            for idx, s in enumerate(flat_steps):
                s["advantage"] = adv[idx : idx + 1]
        return flat_steps

    @torch.no_grad()
    def generate_paths(self, loader: DataLoader, greedy: bool = True) -> List[Dict[str, Any]]:
        self.model.eval()
        results: List[Dict[str, Any]] = []
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            histories = [row[:l].tolist() for row, l in zip(batch["history"], batch["lengths"])]
            corrects = [row[:l].tolist() for row, l in zip(batch["history_correct"], batch["lengths"])]
            deltas = [row[:l].tolist() for row, l in zip(batch["history_deltas"], batch["lengths"])]
            targets = batch["target"].tolist()
            paths = [[] for _ in histories]
            for _ in range(self.path_len):
                state_batch = lists_to_batch(histories, corrects, deltas, targets, self.device)
                action_mask = self._candidate_mask(histories, corrects, targets)
                out = self.model.act(
                    self.graph,
                    state_batch["history"],
                    state_batch["history_correct"],
                    state_batch["history_deltas"],
                    state_batch["lengths"],
                    state_batch["target"],
                    action_mask,
                    greedy=greedy,
                )
                actions = out["action"].detach().cpu().tolist()
                next_corrects = []
                for i, action in enumerate(actions):
                    reward, next_correct = self._step_reward(histories[i], corrects[i], deltas[i], action, targets[i], path_done=False)
                    _ = reward
                    next_corrects.append(next_correct)
                    paths[i].append(int(action))
                histories = [h + [a] for h, a in zip(histories, actions)]
                corrects = [c + [nc] for c, nc in zip(corrects, next_corrects)]
                deltas = [d + [d[-1] if d else 0.0] for d in deltas]
            for i in range(len(paths)):
                results.append(
                    {
                        "user_id": batch["user_id"][i],
                        "history": batch["history"][i, : batch["lengths"][i]].detach().cpu().tolist(),
                        "target": targets[i],
                        "path": paths[i],
                        "future": batch["future"][i, : batch["future_lengths"][i]].detach().cpu().tolist(),
                        "history_correct": batch["history_correct"][i, : batch["lengths"][i]].detach().cpu().tolist(),
                        "history_deltas": batch["history_deltas"][i, : batch["lengths"][i]].detach().cpu().tolist(),
                    }
                )
        return results

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, greedy: bool = True) -> Dict[str, float]:
        predictions = self.generate_paths(loader, greedy=greedy)
        rows = []
        for row in predictions:
            rows.append(
                {
                    "mastery_gain": mastery_gain_metric(
                        self.kt_model,
                        row["history"],
                        row["history_correct"],
                        row["history_deltas"],
                        row["path"],
                        row["target"],
                        self.device,
                    ),
                    "hit_rate": hit_rate(row["path"], row["target"]),
                    "ndcg@10": ndcg_at_k(row["path"], row["future"], 10),
                    "mrr": mrr(row["path"], row["future"]),
                    "prereq_violation": prerequisite_violation_rate(self.graph, row["history"], row["path"]),
                    "difficulty_smoothness": difficulty_smoothness(self.graph, row["history"], row["path"]),
                    "review_coverage": review_coverage(self.graph, row["target"], row["path"]),
                }
            )
        return aggregate(rows)



def make_kt_loaders(
    dataset: StandardDataset,
    batch_size: int = 64,
    max_seq_len: int = 100,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
):
    from .data import KTSequenceDataset

    device = device or torch.device("cpu")
    loader_kwargs = dataloader_kwargs(device, num_workers=num_workers, pin_memory=pin_memory)
    train = KTSequenceDataset(dataset.train_sequences, max_seq_len=max_seq_len)
    val = KTSequenceDataset(dataset.val_sequences, max_seq_len=max_seq_len)
    test = KTSequenceDataset(dataset.test_sequences, max_seq_len=max_seq_len)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_kt, **loader_kwargs),
        DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_kt, **loader_kwargs),
        DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_kt, **loader_kwargs),
    )


def make_task_loaders(
    dataset: StandardDataset,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
):
    from .data import LPRTaskDataset

    device = device or torch.device("cpu")
    loader_kwargs = dataloader_kwargs(device, num_workers=num_workers, pin_memory=pin_memory)
    train = LPRTaskDataset(dataset.train_tasks)
    val = LPRTaskDataset(dataset.val_tasks)
    test = LPRTaskDataset(dataset.test_tasks)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_tasks, **loader_kwargs),
        DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_tasks, **loader_kwargs),
        DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_tasks, **loader_kwargs),
    )
