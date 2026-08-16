#!/usr/bin/env python3
"""One-file ImageNet experiment: CSR (InfoNCE) versus CSR (partial M3G).

Pipeline
--------
1. Download/cache torchvision's pretrained ResNet-18 weights locally.
2. Freeze the backbone and cache deterministic ImageNet train/validation features.
3. Train two identically initialized tied Top-K sparse autoencoders:
   - ``csr``: reconstruction + cross-sparsity non-negative InfoNCE.
   - ``mmpot``: reconstruction + multimarginal partial matching gap (M3PG).
4. Encode the train split as the gallery and validation split as queries.
5. Evaluate exact L2 1-NN with FAISS ``IndexFlatL2`` at every requested Top-K.
6. Save checkpoints, histories, JSON results, and a comparison CSV.

The MMPOT arm treats Top-K, Top-2K, and Top-4K codes of the same frozen image
embedding as three aligned views.  Its multiway cost is the circular-variance
cost of Piran et al. (2024), and its reference partial polymatching is ``s J``.
This is a research extension, not a claim that it appeared in either source
paper.

Expected ImageNet layout
------------------------
    /path/to/imagenet/
      train/n01440764/*.JPEG
      ...
      val/n01440764/*.JPEG
      ...

Example lightweight development run
-----------------------------------
    python csr_vs_mmpot_imagenet.py --data-root /path/to/imagenet \
      --cache-dir runs/csr_mmpot/cache --output-dir runs/csr_mmpot \
      --max-train 50000 --max-val 10000 --epochs 3 --batch-size 256

Paper-scale data (resource intensive)
-------------------------------------
    python csr_vs_mmpot_imagenet.py --data-root /path/to/imagenet \
      --cache-dir /fastssd/imagenet_rn18 --output-dir runs/csr_mmpot \
      --epochs 10 --batch-size 4096 --hidden-dim 2048 \
      --topk 8,16,32,64,128,256 --amp

Dependencies: torch, torchvision, numpy, and faiss-cpu (or faiss-gpu).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models


METHODS = ("csr", "mmpot")


def parse_ints(value: str) -> List[int]:
    try:
        result = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return sorted(set(result))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Frozen RN18 + Top-K SAE: CSR InfoNCE versus partial multimarginal matching gap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("ImageNet and feature cache")
    data.add_argument("--data-root", type=Path, required=True, help="ImageFolder root or Hugging Face cache directory")
    data.add_argument("--data-backend", choices=("imagefolder", "hf"), default="imagefolder")
    data.add_argument("--hf-dataset-id", default="ILSVRC/imagenet-1k")
    data.add_argument("--hf-revision", default="main")
    data.add_argument("--hf-token-env", default="HF_TOKEN")
    data.add_argument("--cache-dir", type=Path, default=Path("runs/csr_mmpot/cache"))
    data.add_argument("--weights-cache", type=Path, default=Path("weights"), help="local TORCH_HOME for RN18 weights")
    data.add_argument("--rebuild-cache", action="store_true")
    data.add_argument("--feature-batch-size", type=int, default=512)
    data.add_argument("--workers", type=int, default=8)
    data.add_argument("--max-train", type=int, default=0, help="deterministic subset; 0 uses all training images")
    data.add_argument("--max-val", type=int, default=0, help="deterministic subset; 0 uses all validation images")

    model = p.add_argument_group("sparse autoencoder")
    model.add_argument("--hidden-dim", type=int, default=2048, help="paper default rule h=4d for RN18 d=512")
    model.add_argument("--topk", type=parse_ints, default=[8, 16, 32, 64, 128, 256])
    model.add_argument("--train-k", type=int, default=32)
    model.add_argument("--k-aux", type=int, default=512)
    model.add_argument("--aux-weight", type=float, default=1.0 / 32.0)
    model.add_argument("--dead-steps", type=int, default=1000)
    model.add_argument("--multi-topk-weight", type=float, default=1.0 / 8.0)

    train = p.add_argument_group("training")
    train.add_argument("--method", choices=("csr", "mmpot", "both"), default="both")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--lr", type=float, default=4e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--contrast-weight", type=float, default=0.1, help="gamma for CSR/NCL, matching the vision setup")
    train.add_argument("--temperature", type=float, default=0.2)
    train.add_argument("--mmpot-weight", type=float, default=0.1)
    train.add_argument("--ot-mass", type=float, default=0.8)
    train.add_argument("--ot-eta", type=float, default=0.2, help="M3G recommendation for circular variance")
    train.add_argument("--ot-iters", type=int, default=100)
    train.add_argument("--ot-tol", type=float, default=1e-4)
    train.add_argument("--ot-microbatch", type=int, default=32, help="B^3 cost makes small OT groups necessary")
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--print-freq", type=int, default=50)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--output-dir", type=Path, default=Path("runs/csr_mmpot"))

    knn = p.add_argument_group("exact FAISS 1-NN")
    knn.add_argument("--knn-batch-size", type=int, default=4096)
    knn.add_argument("--knn-query-batch", type=int, default=4096)
    knn.add_argument("--knn-normalize", action=argparse.BooleanOptionalAction, default=False)
    knn.add_argument("--faiss-gpu", action="store_true", help="use GPU IndexFlatL2 if faiss-gpu is installed")
    return p


def validate_args(a: argparse.Namespace) -> None:
    if a.data_backend == "imagefolder":
        for split in ("train", "val"):
            if not (a.data_root / split).is_dir():
                raise FileNotFoundError(f"missing ImageNet directory: {a.data_root / split}")
    elif not a.data_root.is_dir():
        raise FileNotFoundError(f"missing Hugging Face cache directory: {a.data_root}")
    if a.hidden_dim < max(max(a.topk), 4 * a.train_k):
        raise ValueError("hidden-dim must be >= max(topk) and >= 4*train-k")
    if not 0.0 < a.ot_mass <= 1.0:
        raise ValueError("ot-mass must be in (0,1]")
    if min(a.epochs, a.batch_size, a.feature_batch_size, a.ot_microbatch) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if a.ot_microbatch < 2:
        raise ValueError("ot-microbatch must be at least 2")


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class FrozenResNet18(nn.Module):
    def __init__(self, weights_cache: Path) -> None:
        super().__init__()
        # torchvision downloads the official weight once into TORCH_HOME/hub/checkpoints.
        os.environ["TORCH_HOME"] = str(weights_cache.expanduser().resolve())
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        network = models.resnet18(weights=weights)
        network.fc = nn.Identity()
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self.network = network
        self.transform = weights.transforms()
        self.output_dim = 512

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class HuggingFaceImages(Dataset):
    """Apply torchvision preprocessing to a cached Hugging Face split."""

    def __init__(self, split: Any, transform: Any) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.split[index]
        image = sample["image"]
        if image is None:
            raise RuntimeError(f"ImageNet sample {index} has no decoded image")
        return self.transform(image.convert("RGB")), int(sample["label"])


def deterministic_subset(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def cache_paths(cache_dir: Path, split: str) -> Tuple[Path, Path, Path]:
    return (
        cache_dir / f"{split}_features.f16.npy",
        cache_dir / f"{split}_labels.npy",
        cache_dir / f"{split}_meta.json",
    )


@torch.inference_mode()
def cache_split(
    split: str,
    root: Path,
    backbone: FrozenResNet18,
    device: torch.device,
    cache_dir: Path,
    batch_size: int,
    workers: int,
    maximum: int,
    seed: int,
    rebuild: bool,
    data_backend: str,
    hf_dataset_id: str,
    hf_revision: str,
    hf_token_env: str,
) -> Dict[str, Any]:
    feature_path, label_path, meta_path = cache_paths(cache_dir, split)
    if not rebuild and feature_path.is_file() and label_path.is_file() and meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if data_backend == "imagefolder":
        dataset: Dataset = datasets.ImageFolder(root / split, transform=backbone.transform)
        source_split = split
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Hugging Face backend requires: pip install datasets") from exc
        source_split = "validation" if split == "val" else "train"
        token_value = os.environ.get(hf_token_env)
        hf_split = load_dataset(
            path=hf_dataset_id,
            split=source_split,
            cache_dir=str(root),
            revision=hf_revision,
            token=token_value if token_value else True,
        )
        dataset = HuggingFaceImages(hf_split, backbone.transform)
    dataset = deterministic_subset(dataset, maximum, seed + (0 if split == "train" else 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16, shape=(len(dataset), backbone.output_dim)
    )
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64, shape=(len(dataset),))
    backbone.eval()
    offset, started = 0, time.time()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            batch_features = backbone(images).float()
        count = images.shape[0]
        features[offset : offset + count] = batch_features.cpu().numpy().astype(np.float16)
        labels[offset : offset + count] = target.numpy()
        offset += count
        if offset % (batch_size * 100) < count:
            print(f"cache {split}: {offset}/{len(dataset)}", flush=True)
    features.flush()
    labels.flush()
    metadata = {
        "split": split,
        "samples": len(dataset),
        "feature_dim": backbone.output_dim,
        "dtype": "float16",
        "backbone": "torchvision_resnet18_IMAGENET1K_V1",
        "data_backend": data_backend,
        "source_split": source_split,
        "seconds": time.time() - started,
    }
    atomic_json(metadata, meta_path)
    return metadata


class CachedFeatures(Dataset):
    def __init__(self, feature_path: Path, label_path: Path) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if len(self.features) != len(self.labels):
            raise RuntimeError("feature and label caches have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Copy avoids PyTorch's warning about read-only numpy memmaps.
        return torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)), int(self.labels[index])


class TopKSAE(nn.Module):
    """Tied Top-K sparse autoencoder with dead-latent auxiliary tracking."""

    def __init__(self, input_dim: int, hidden_dim: int, dead_steps: int) -> None:
        super().__init__()
        decoder = torch.empty(hidden_dim, input_dim)
        nn.init.kaiming_uniform_(decoder, a=math.sqrt(5))
        decoder = F.normalize(decoder, dim=1)
        self.decoder = nn.Parameter(decoder)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("inactive_steps", torch.zeros(hidden_dim, dtype=torch.long))
        self.dead_steps = int(dead_steps)

    @staticmethod
    def keep_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, pre.shape[1])
        values, indices = torch.topk(pre, k=k, dim=1, sorted=False)
        values = F.relu(values)
        result = torch.zeros_like(pre)
        return result.scatter(1, indices, values)

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pre_bias) @ self.decoder.T + self.encoder_bias

    def encode(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self.keep_topk(self.preactivations(x), k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.decoder + self.pre_bias

    def update_activity(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            active = z.gt(0).any(dim=0)
            self.inactive_steps.add_(1)
            self.inactive_steps[active] = 0

    def reconstruction_losses(
        self, x: torch.Tensor, k: int, k_aux: int, multi_weight: float, aux_weight: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pre = self.preactivations(x)
        z1 = self.keep_topk(pre, k)
        z2 = self.keep_topk(pre, min(2 * k, self.decoder.shape[0]))
        z4 = self.keep_topk(pre, min(4 * k, self.decoder.shape[0]))
        recon1 = self.decode(z1)
        recon4 = self.decode(z4)
        main = F.mse_loss(recon1, x)
        multi = F.mse_loss(recon4, x)

        dead = self.inactive_steps >= self.dead_steps
        if dead.any() and k_aux > 0:
            masked = pre.masked_fill(~dead[None, :], -torch.inf)
            aux_k = min(k_aux, int(dead.sum()))
            aux_z = self.keep_topk(masked, aux_k)
            residual = (x - recon1).detach()
            aux = F.mse_loss(aux_z @ self.decoder, residual)
        else:
            aux = main.new_zeros(())
        total = main + multi_weight * multi + aux_weight * aux
        self.update_activity(z1)
        stats = {"recon": main, "multi_recon": multi, "aux": aux, "dead_fraction": dead.float().mean()}
        return total, stats, (z1, z2, z4)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.copy_(F.normalize(self.decoder, dim=1))


def cross_view_infonce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE; the same image at two sparsity levels is positive."""
    a = F.normalize(z_a, dim=1, eps=1e-8)
    b = F.normalize(z_b, dim=1, eps=1e-8)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def circular_variance_cost(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
    """Exact M3G circular variance for three unit-normalized representation views."""
    a = F.normalize(z1.float(), dim=1, eps=1e-8)
    b = F.normalize(z2.float(), dim=1, eps=1e-8)
    c = F.normalize(z3.float(), dim=1, eps=1e-8)
    d12 = 1.0 - a @ b.T
    d13 = 1.0 - a @ c.T
    d23 = 1.0 - b @ c.T
    return ((2.0 / 9.0) * (d12[:, :, None] + d13[:, None, :] + d23[None, :, :])).clamp(0.0, 1.0)


def kl_constraint(target: torch.Tensor, value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    t = target.clamp_min(eps)
    v = value.clamp_min(eps)
    return (v - target + t * torch.log(t / v)).sum()


@torch.no_grad()
def greenkhorn_mmpot(
    cost: torch.Tensor, mass: float, eta: float, max_iters: int, tol: float
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, float]]:
    """Greedy 3-marginal partial OT solver from the slack-variable dual."""
    if cost.ndim != 3 or len(set(cost.shape)) != 1:
        raise ValueError("expected a cubic [B,B,B] cost tensor")
    n = cost.shape[0]
    p = torch.full((n,), 1.0 / n, device=cost.device, dtype=torch.float32)
    kernel = torch.exp((-cost.float() / eta).clamp(min=-60.0, max=0.0))
    scales = [torch.ones_like(p) for _ in range(3)]
    w = cost.new_ones((), dtype=torch.float32)
    target_mass = cost.new_tensor(mass, dtype=torch.float32)
    D = mass + 3.0 * (1.0 - mass)
    error = float("inf")

    for iteration in range(max_iters):
        v1, v2, v3 = scales
        B = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
        partition = (w * B.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(1e-12)
        factor = D / partition
        estimates = [
            factor * (w * B.sum(dim=(1, 2)) + v1),
            factor * (w * B.sum(dim=(0, 2)) + v2),
            factor * (w * B.sum(dim=(0, 1)) + v3),
        ]
        current_mass = factor * w * B.sum()
        errors = torch.stack(
            [kl_constraint(p, r) for r in estimates]
            + [kl_constraint(target_mass[None], current_mass[None])]
        )
        error = float(errors.max())
        if error <= tol:
            break
        worst = int(errors.argmax())
        if worst < 3:
            scales[worst].mul_(p / estimates[worst].clamp_min(1e-12))
            scales[worst].clamp_(1e-12, 1e12)
        else:
            w.mul_(target_mass / current_mass.clamp_min(1e-12)).clamp_(1e-12, 1e12)

    v1, v2, v3 = scales
    B = kernel * v1[:, None, None] * v2[None, :, None] * v3[None, None, :]
    partition = (w * B.sum() + v1.sum() + v2.sum() + v3.sum()).clamp_min(1e-12)
    factor = D / partition
    plan = factor * w * B
    marginals = [plan.sum((1, 2)), plan.sum((0, 2)), plan.sum((0, 1))]
    slacks = [(p - r).clamp_min(0.0) for r in marginals]
    cap = max(float((r - p).clamp_min(0).max()) for r in marginals)
    return plan, slacks, {
        "mass": float(plan.sum()),
        "mass_error": abs(float(plan.sum()) - mass),
        "cap_violation": cap,
        "constraint_error": error,
        "iterations": iteration + 1,
    }


def entropy_term(x: torch.Tensor) -> torch.Tensor:
    positive = x > 0
    return (x[positive] * (torch.log(x[positive]) - 1.0)).sum()


def partial_matching_gap(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    mass: float,
    eta: float,
    max_iters: int,
    tol: float,
    microbatch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Partial M3G, averaged over independent OT microbatches.

    The solver is detached (Danskin/envelope differentiation).  The returned
    scalar has the true regularized gap value while its gradient is sJ-X*.
    """
    losses: List[torch.Tensor] = []
    diagnostics: List[Dict[str, float]] = []
    for start in range(0, z1.shape[0], microbatch):
        stop = min(start + microbatch, z1.shape[0])
        if stop - start < 2:
            continue
        cost = circular_variance_cost(z1[start:stop], z2[start:stop], z3[start:stop])
        with torch.no_grad():
            plan, slacks, diag = greenkhorn_mmpot(cost.detach(), mass, eta, max_iters, tol)
        n = cost.shape[0]
        diagonal = torch.arange(n, device=cost.device)
        reference_transport = mass * cost[diagonal, diagonal, diagonal].mean()
        optimum_transport = (cost * plan).sum()
        gradient_gap = reference_transport - optimum_transport

        # Entropic values make this the actual optimality gap. They are detached;
        # the envelope gradient through C remains sJ-X*.
        p = cost.new_full((n,), 1.0 / n)
        ref_plan_values = cost.new_full((n,), mass / n)
        ref_slack = (1.0 - mass) * p
        ref_entropy = entropy_term(ref_plan_values) + 3.0 * entropy_term(ref_slack)
        opt_entropy = entropy_term(plan) + sum(entropy_term(q) for q in slacks)
        true_value = gradient_gap.detach() + eta * (ref_entropy - opt_entropy)
        loss = gradient_gap + (true_value - gradient_gap.detach())
        losses.append(loss)
        diagnostics.append(diag)
    if not losses:
        raise RuntimeError("no valid MMPOT microbatch")
    summary = {
        key: sum(item[key] for item in diagnostics) / len(diagnostics)
        for key in diagnostics[0]
    }
    return torch.stack(losses).mean(), summary


@dataclass
class EpochResult:
    total: float
    reconstruction: float
    representation: float
    dead_fraction: float
    ot_mass_error: float
    seconds: float


def save_checkpoint(
    path: Path, model: TopKSAE, optimizer: torch.optim.Optimizer, epoch: int, history: List[Dict[str, Any]], args: argparse.Namespace
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history, "args": vars(args)},
        tmp,
    )
    os.replace(tmp, path)


def train_method(
    method: str,
    initial_state: Mapping[str, torch.Tensor],
    dataset: CachedFeatures,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[TopKSAE, List[Dict[str, Any]]]:
    model = TopKSAE(512, args.hidden_dim, args.dead_steps).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=6.25e-10)
    method_dir = args.output_dir / method
    checkpoint = method_dir / "last.pt"
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    if args.resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state.get("history", [])
        start_epoch = int(state["epoch"]) + 1

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        sums = {"total": 0.0, "recon": 0.0, "repr": 0.0, "dead": 0.0, "mass_error": 0.0}
        samples, started = 0, time.time()
        for step, (features, _) in enumerate(loader):
            features = features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                recon_loss, recon_stats, views = model.reconstruction_losses(
                    features, args.train_k, args.k_aux, args.multi_topk_weight, args.aux_weight
                )
                if method == "csr":
                    repr_loss = cross_view_infonce(views[0], views[2], args.temperature)
                    objective = recon_loss + args.contrast_weight * repr_loss
                    mass_error = 0.0
                else:
                    # Force the numerically sensitive OT path to float32.
                    with torch.autocast(device_type=device.type, enabled=False):
                        repr_loss, ot_diag = partial_matching_gap(
                            views[0].float(), views[1].float(), views[2].float(),
                            args.ot_mass, args.ot_eta, args.ot_iters, args.ot_tol, args.ot_microbatch,
                        )
                    objective = recon_loss + args.mmpot_weight * repr_loss
                    mass_error = ot_diag["mass_error"]
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
            model.normalize_decoder()
            count = features.shape[0]
            samples += count
            sums["total"] += float(objective.detach()) * count
            sums["recon"] += float(recon_loss.detach()) * count
            sums["repr"] += float(repr_loss.detach()) * count
            sums["dead"] += float(recon_stats["dead_fraction"]) * count
            sums["mass_error"] += mass_error * count
            if step % args.print_freq == 0:
                print(
                    f"{method} epoch={epoch+1} step={step}/{len(loader)} "
                    f"loss={float(objective):.5f} recon={float(recon_loss):.5f} repr={float(repr_loss):.5f}",
                    flush=True,
                )
        result = EpochResult(
            total=sums["total"] / samples,
            reconstruction=sums["recon"] / samples,
            representation=sums["repr"] / samples,
            dead_fraction=sums["dead"] / samples,
            ot_mass_error=sums["mass_error"] / samples,
            seconds=time.time() - started,
        )
        history.append({"epoch": epoch + 1, **asdict(result)})
        save_checkpoint(checkpoint, model, optimizer, epoch, history, args)
        atomic_json({"method": method, "history": history}, method_dir / "history.json")
    return model, history


@torch.inference_mode()
def add_gallery_to_faiss(
    index: Any, model: TopKSAE, dataset: CachedFeatures, k: int, batch_size: int, device: torch.device, normalize: bool
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels: List[np.ndarray] = []
    model.eval()
    for features, target in loader:
        z = model.encode(features.to(device), k).float()
        if normalize:
            z = F.normalize(z, dim=1)
        index.add(np.ascontiguousarray(z.cpu().numpy(), dtype=np.float32))
        labels.append(target.numpy())
    return np.concatenate(labels)


@torch.inference_mode()
def search_queries(
    index: Any,
    gallery_labels: np.ndarray,
    model: TopKSAE,
    dataset: CachedFeatures,
    k: int,
    batch_size: int,
    device: torch.device,
    normalize: bool,
) -> Tuple[float, float, int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, total, distance_sum = 0, 0, 0.0
    model.eval()
    for features, target in loader:
        z = model.encode(features.to(device), k).float()
        if normalize:
            z = F.normalize(z, dim=1)
        distances, indices = index.search(np.ascontiguousarray(z.cpu().numpy(), dtype=np.float32), 1)
        predictions = gallery_labels[indices[:, 0]]
        truth = target.numpy()
        correct += int((predictions == truth).sum())
        total += len(truth)
        distance_sum += float(distances[:, 0].sum())
    return 100.0 * correct / total, distance_sum / total, total


def make_faiss_index(dimension: int, use_gpu: bool) -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required. Install with: pip install faiss-cpu") from exc
    cpu_index = faiss.IndexFlatL2(dimension)
    if not use_gpu:
        return cpu_index
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("--faiss-gpu requested, but the installed FAISS has no GPU support")
    resources = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(resources, 0, cpu_index)


def benchmark_method(
    method: str,
    model: TopKSAE,
    train_data: CachedFeatures,
    val_data: CachedFeatures,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for k in args.topk:
        print(f"FAISS exact L2: method={method} k={k}", flush=True)
        index = make_faiss_index(args.hidden_dim, args.faiss_gpu)
        gallery_labels = add_gallery_to_faiss(
            index, model, train_data, k, args.knn_batch_size, device, args.knn_normalize
        )
        accuracy, mean_distance, queries = search_queries(
            index, gallery_labels, model, val_data, k, args.knn_query_batch, device, args.knn_normalize
        )
        results[str(k)] = {
            "top1": accuracy,
            "mean_neighbor_l2_squared": mean_distance,
            "gallery_samples": len(gallery_labels),
            "query_samples": queries,
        }
        del index
    return {
        "protocol": "FAISS_IndexFlatL2_train_gallery_validation_queries_1NN",
        "normalized": args.knn_normalize,
        "per_topk": results,
    }


def write_comparison(results: Mapping[str, Any], path: Path) -> None:
    if not all(method in results for method in METHODS):
        return
    rows = []
    for k, csr in results["csr"]["knn"]["per_topk"].items():
        other = results["mmpot"]["knn"]["per_topk"][k]
        rows.append({
            "topk": int(k),
            "csr_1nn_top1": csr["top1"],
            "mmpot_1nn_top1": other["top1"],
            "delta_mmpot_minus_csr": other["top1"] - csr["top1"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.weights_cache = args.weights_cache.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = choose_device(args.device)
    print(f"device={device} output={args.output_dir}", flush=True)

    backbone = FrozenResNet18(args.weights_cache).to(device)
    train_meta = cache_split(
        "train", args.data_root, backbone, device, args.cache_dir, args.feature_batch_size,
        args.workers, args.max_train, args.seed, args.rebuild_cache,
        args.data_backend, args.hf_dataset_id, args.hf_revision, args.hf_token_env,
    )
    val_meta = cache_split(
        "val", args.data_root, backbone, device, args.cache_dir, args.feature_batch_size,
        args.workers, args.max_val, args.seed, args.rebuild_cache,
        args.data_backend, args.hf_dataset_id, args.hf_revision, args.hf_token_env,
    )
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features, train_labels, _ = cache_paths(args.cache_dir, "train")
    val_features, val_labels, _ = cache_paths(args.cache_dir, "val")
    train_data = CachedFeatures(train_features, train_labels)
    val_data = CachedFeatures(val_features, val_labels)

    seed_all(args.seed)
    template = TopKSAE(512, args.hidden_dim, args.dead_steps)
    # Initialize pre-bias to the cached feature mean, as in SAE practice.
    sample_count = min(len(train_data.features), 100_000)
    template.pre_bias.data.copy_(
        torch.from_numpy(np.asarray(train_data.features[:sample_count], dtype=np.float32).mean(axis=0))
    )
    initial_state = {key: value.clone() for key, value in template.state_dict().items()}
    methods = list(METHODS) if args.method == "both" else [args.method]
    results: Dict[str, Any] = {}
    for method in methods:
        seed_all(args.seed)
        model, history = train_method(method, initial_state, train_data, device, args)
        knn = benchmark_method(method, model, train_data, val_data, device, args)
        results[method] = {"history": history, "knn": knn}
        atomic_json(results[method], args.output_dir / method / "results.json")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "experiment": "CSR_InfoNCE_vs_CSR_Multimarginal_Partial_Matching_Gap",
        "backbone": "frozen_torchvision_resnet18_IMAGENET1K_V1",
        "dataset": {"train": train_meta, "validation": val_meta},
        "config": serializable_args(args),
        "results": results,
    }
    atomic_json(summary, args.output_dir / "summary.json")
    write_comparison(results, args.output_dir / "comparison.csv")
    print(f"complete: {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
