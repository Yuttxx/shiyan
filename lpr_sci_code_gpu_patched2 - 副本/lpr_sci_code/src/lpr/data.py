from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .common import load_json, pad_float_sequences, pad_sequences, read_jsonl, save_json, write_jsonl


@dataclass
class GraphStore:
    num_nodes: int
    relations: List[str]
    edges_by_rel: Dict[str, List[Tuple[int, int, float]]]
    prerequisites: Dict[int, List[int]]
    successors: Dict[int, List[int]]
    similarity: Dict[int, List[Tuple[int, float]]]
    difficulty: Dict[int, float]
    concept_meta: Dict[str, Any]

    @property
    def num_relations(self) -> int:
        return len(self.relations)


@dataclass
class StandardDataset:
    concept2id: Dict[str, int]
    id2concept: List[str]
    graph: GraphStore
    train_sequences: List[Dict[str, Any]]
    val_sequences: List[Dict[str, Any]]
    test_sequences: List[Dict[str, Any]]
    train_tasks: List[Dict[str, Any]]
    val_tasks: List[Dict[str, Any]]
    test_tasks: List[Dict[str, Any]]


def normalize_time_deltas(timestamps: Sequence[float]) -> List[float]:
    if len(timestamps) == 0:
        return []
    deltas = [0.0]
    for i in range(1, len(timestamps)):
        gap = max(float(timestamps[i]) - float(timestamps[i - 1]), 0.0)
        deltas.append(math.log1p(gap))
    return deltas


class KTSequenceDataset(Dataset):
    def __init__(self, sequences: List[Dict[str, Any]], max_seq_len: int = 200):
        self.samples: List[Dict[str, Any]] = []
        for row in sequences:
            concepts = row["concepts"]
            correct = row["correct"]
            deltas = row.get("deltas", [0.0] * len(concepts))
            if len(concepts) < 2:
                continue
            for end in range(2, len(concepts) + 1):
                start = max(0, end - max_seq_len)
                self.samples.append(
                    {
                        "concepts": concepts[start:end],
                        "correct": correct[start:end],
                        "deltas": deltas[start:end],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        return row


class LPRTaskDataset(Dataset):
    def __init__(self, tasks: List[Dict[str, Any]]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.tasks[idx]


class GRU4RecDataset(Dataset):
    def __init__(self, tasks: List[Dict[str, Any]]):
        self.tasks = [t for t in tasks if len(t.get("future", [])) > 0]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        return {
            "history": task["history"],
            "correct": task.get("history_correct", [1] * len(task["history"])),
            "deltas": task.get("history_deltas", [0.0] * len(task["history"])),
            "label": task["future"][0],
            "target": task["target"],
        }


def collate_kt(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    concepts = [row["concepts"][:-1] for row in batch]
    correct = [row["correct"][:-1] for row in batch]
    deltas = [row.get("deltas", [0.0] * len(row["concepts"]))[:-1] for row in batch]
    target = [row["concepts"][1:] for row in batch]
    target_correct = [row["correct"][1:] for row in batch]
    lengths = torch.tensor([len(x) for x in concepts], dtype=torch.long)
    return {
        "concepts": pad_sequences(concepts, pad_value=0),
        "correct": pad_float_sequences(correct, pad_value=-1.0),
        "deltas": pad_float_sequences(deltas, pad_value=0.0),
        "target_concepts": pad_sequences(target, pad_value=0),
        "target_correct": pad_float_sequences(target_correct, pad_value=-1.0),
        "lengths": lengths,
    }


def collate_tasks(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    history = [row["history"] for row in batch]
    hist_correct = [row.get("history_correct", [1] * len(row["history"])) for row in batch]
    hist_deltas = [row.get("history_deltas", [0.0] * len(row["history"])) for row in batch]
    lengths = torch.tensor([len(x) for x in history], dtype=torch.long)
    future = [row.get("future", []) for row in batch]
    future_pad = pad_sequences([f if f else [0] for f in future], pad_value=0)
    future_len = torch.tensor([len(f) for f in future], dtype=torch.long)
    return {
        "history": pad_sequences(history, pad_value=0),
        "history_correct": pad_float_sequences(hist_correct, pad_value=-1.0),
        "history_deltas": pad_float_sequences(hist_deltas, pad_value=0.0),
        "lengths": lengths,
        "target": torch.tensor([row["target"] for row in batch], dtype=torch.long),
        "future": future_pad,
        "future_lengths": future_len,
        "user_id": [row["user_id"] for row in batch],
    }



def collate_gru4rec(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    history = [row["history"] for row in batch]
    correct = [row.get("correct", [1] * len(row["history"])) for row in batch]
    deltas = [row.get("deltas", [0.0] * len(row["history"])) for row in batch]
    lengths = torch.tensor([len(x) for x in history], dtype=torch.long)
    return {
        "history": pad_sequences(history, pad_value=0),
        "correct": pad_float_sequences(correct, pad_value=-1.0),
        "deltas": pad_float_sequences(deltas, pad_value=0.0),
        "lengths": lengths,
        "label": torch.tensor([row["label"] for row in batch], dtype=torch.long),
        "target": torch.tensor([row["target"] for row in batch], dtype=torch.long),
    }

@dataclass
class SequenceBuildConfig:
    min_user_interactions: int = 20
    history_len: int = 20
    path_len: int = 10
    stride: int = 5



def build_tasks_from_sequences(
    sequences: List[Dict[str, Any]],
    config: SequenceBuildConfig,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for row in sequences:
        concepts = row["concepts"]
        correct = row["correct"]
        timestamps = row["timestamps"]
        if len(concepts) < config.history_len + config.path_len:
            continue
        deltas = row.get("deltas", normalize_time_deltas(timestamps))
        max_start = len(concepts) - (config.history_len + config.path_len) + 1
        for start in range(0, max_start, config.stride):
            hist_start = start
            hist_end = start + config.history_len
            fut_end = hist_end + config.path_len
            future = concepts[hist_end:fut_end]
            if not future:
                continue
            tasks.append(
                {
                    "user_id": row["user_id"],
                    "history": concepts[hist_start:hist_end],
                    "history_correct": correct[hist_start:hist_end],
                    "history_timestamps": timestamps[hist_start:hist_end],
                    "history_deltas": deltas[hist_start:hist_end],
                    "target": future[-1],
                    "future": future,
                    "future_correct": correct[hist_end:fut_end],
                }
            )
    return tasks



def split_by_user(sequences: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
    rng = np.random.default_rng(seed)
    user_ids = np.array(sorted({row["user_id"] for row in sequences}))
    rng.shuffle(user_ids)
    n = len(user_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_users = set(user_ids[:n_train])
    val_users = set(user_ids[n_train : n_train + n_val])
    test_users = set(user_ids[n_train + n_val :])
    train = [row for row in sequences if row["user_id"] in train_users]
    val = [row for row in sequences if row["user_id"] in val_users]
    test = [row for row in sequences if row["user_id"] in test_users]
    return train, val, test



def save_standard_dataset(
    output_dir: str | Path,
    concept2id: Dict[str, int],
    graph: GraphStore,
    train_sequences: List[Dict[str, Any]],
    val_sequences: List[Dict[str, Any]],
    test_sequences: List[Dict[str, Any]],
    train_tasks: List[Dict[str, Any]],
    val_tasks: List[Dict[str, Any]],
    test_tasks: List[Dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_id = max(concept2id.values()) if concept2id else -1
    id2concept = [None] * (max_id + 1)
    for k, v in concept2id.items():
        id2concept[v] = k
    save_json(concept2id, output_dir / "concept2id.json")
    save_json(id2concept, output_dir / "id2concept.json")
    save_json(
        {
            "num_nodes": graph.num_nodes,
            "relations": graph.relations,
            "edges_by_rel": {
                rel: [[int(s), int(d), float(w)] for s, d, w in edges]
                for rel, edges in graph.edges_by_rel.items()
            },
            "prerequisites": {str(k): v for k, v in graph.prerequisites.items()},
            "successors": {str(k): v for k, v in graph.successors.items()},
            "similarity": {str(k): [[int(n), float(w)] for n, w in v] for k, v in graph.similarity.items()},
            "difficulty": {str(k): float(v) for k, v in graph.difficulty.items()},
            "concept_meta": graph.concept_meta,
        },
        output_dir / "graph.json",
    )
    write_jsonl(train_sequences, output_dir / "train_sequences.jsonl")
    write_jsonl(val_sequences, output_dir / "val_sequences.jsonl")
    write_jsonl(test_sequences, output_dir / "test_sequences.jsonl")
    write_jsonl(train_tasks, output_dir / "train_tasks.jsonl")
    write_jsonl(val_tasks, output_dir / "val_tasks.jsonl")
    write_jsonl(test_tasks, output_dir / "test_tasks.jsonl")



def load_standard_dataset(dataset_dir: str | Path) -> StandardDataset:
    dataset_dir = Path(dataset_dir)
    concept2id = load_json(dataset_dir / "concept2id.json")
    id2concept = load_json(dataset_dir / "id2concept.json")
    g = load_json(dataset_dir / "graph.json")
    graph = GraphStore(
        num_nodes=int(g["num_nodes"]),
        relations=list(g["relations"]),
        edges_by_rel={k: [(int(s), int(d), float(w)) for s, d, w in v] for k, v in g["edges_by_rel"].items()},
        prerequisites={int(k): [int(x) for x in v] for k, v in g["prerequisites"].items()},
        successors={int(k): [int(x) for x in v] for k, v in g["successors"].items()},
        similarity={int(k): [(int(n), float(w)) for n, w in v] for k, v in g["similarity"].items()},
        difficulty={int(k): float(v) for k, v in g["difficulty"].items()},
        concept_meta=g["concept_meta"],
    )
    return StandardDataset(
        concept2id=concept2id,
        id2concept=id2concept,
        graph=graph,
        train_sequences=read_jsonl(dataset_dir / "train_sequences.jsonl"),
        val_sequences=read_jsonl(dataset_dir / "val_sequences.jsonl"),
        test_sequences=read_jsonl(dataset_dir / "test_sequences.jsonl"),
        train_tasks=read_jsonl(dataset_dir / "train_tasks.jsonl"),
        val_tasks=read_jsonl(dataset_dir / "val_tasks.jsonl"),
        test_tasks=read_jsonl(dataset_dir / "test_tasks.jsonl"),
    )



def safe_bool_to_float(values: Sequence[Any]) -> List[float]:
    out = []
    for v in values:
        if pd.isna(v):
            out.append(0.0)
        elif isinstance(v, str):
            out.append(1.0 if v.lower() in {"1", "true", "t", "yes"} else 0.0)
        else:
            out.append(float(bool(v)))
    return out
