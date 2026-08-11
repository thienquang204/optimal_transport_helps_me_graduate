#!/usr/bin/env python3
"""Download and validate ImageNet-1K from Hugging Face.

The ILSVRC ImageNet repository is gated. Before running this script:

1. Accept the access conditions at
   https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Authenticate with ``hf auth login`` or export an approved ``HF_TOKEN``.

By default, all official ``train``, ``validation``, and ``test`` splits are
downloaded. The resulting cache is directly consumable by
``matryoshka_mmpot_experiment.py --dataset imagenet --data-root CACHE``.
Training uses ``train`` and metrics use the labelled ``validation`` split;
the official ``test`` split is retained for prediction/submission workflows
and is allowed to be unlabelled. ImageNet is large, so make sure CACHE is on a
volume with ample free space.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import DownloadConfig, load_dataset


DATASET_PAGE = "https://huggingface.co/datasets/ILSVRC/imagenet-1k"


def parse_splits(value: str) -> List[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"train", "validation", "test"}
    invalid = sorted(set(splits) - allowed)
    if not splits or invalid:
        suffix = f"; invalid values: {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(
            f"expected comma-separated train,validation,test splits{suffix}"
        )
    return list(dict.fromkeys(splits))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download gated ILSVRC/imagenet-1k splits into a reusable Hugging Face cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface"))
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=["train", "validation", "test"],
        help="official splits to cache; test may not contain ground-truth labels",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="environment variable containing the access token; cached `hf auth login` is also supported",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--check-samples",
        type=int,
        default=1,
        help="decode this many samples per split after download; 0 disables decoding checks",
    )
    return parser


def validate_split(split_name: str, dataset: Any, check_samples: int) -> Dict[str, Any]:
    if "image" not in dataset.column_names:
        raise RuntimeError(
            f"split {split_name!r} has columns {dataset.column_names}, expected an image column"
        )
    has_label_column = "label" in dataset.column_names
    labels_available = has_label_column
    checked = min(check_samples, len(dataset))
    for index in range(checked):
        sample = dataset[index]
        image = sample["image"]
        if image is None:
            raise RuntimeError(f"split {split_name!r} image {index} failed to decode")
        if hasattr(image, "load"):
            image.load()
        label = sample.get("label")
        if label is None or (isinstance(label, int) and label < 0):
            labels_available = False
            if split_name != "test":
                raise RuntimeError(
                    f"labelled split {split_name!r} sample {index} has no valid label"
                )
    return {
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "features": str(dataset.features),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "has_label_column": has_label_column,
        "sampled_labels_available": labels_available,
        "decoded_samples_checked": checked,
        "cache_files": [entry.get("filename") for entry in dataset.cache_files],
    }


def write_manifest(cache_dir: Path, payload: Dict[str, Any]) -> Path:
    path = cache_dir / "imagenet_download_manifest.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_retries < 0 or args.check_samples < 0:
        raise ValueError("max-retries and check-samples must be non-negative")
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    token_value = os.environ.get(args.token_env)
    token: Any = token_value if token_value else True
    download_config = DownloadConfig(max_retries=args.max_retries)
    manifest: Dict[str, Any] = {
        "dataset_id": args.dataset_id,
        "dataset_page": DATASET_PAGE,
        "revision": args.revision,
        "cache_dir": str(cache_dir),
        "requested_splits": args.splits,
        "completed_at_utc": None,
        "splits": {},
    }

    print(f"Dataset: {args.dataset_id}@{args.revision}")
    print(f"Cache:   {cache_dir}")
    print(f"Splits:  {', '.join(args.splits)}")
    print("Existing cached shards will be reused.", flush=True)
    try:
        for split_name in args.splits:
            print(f"\nDownloading/preparing split: {split_name}", flush=True)
            split = load_dataset(
                path=args.dataset_id,
                split=split_name,
                cache_dir=str(cache_dir),
                revision=args.revision,
                token=token,
                download_config=download_config,
            )
            manifest["splits"][split_name] = validate_split(
                split_name, split, args.check_samples
            )
            print(
                f"Prepared {split_name}: {manifest['splits'][split_name]['rows']:,} rows",
                flush=True,
            )
    except Exception as exc:
        print(
            "\nImageNet download failed. Confirm that you accepted the dataset terms at\n"
            f"{DATASET_PAGE}\nand authenticated with `hf auth login` or {args.token_env}.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = write_manifest(cache_dir, manifest)
    print(f"\nImageNet cache is ready. Manifest: {manifest_path}")
    print(
        "Train with:\n"
        "  python matryoshka_mmpot_experiment.py "
        f"--dataset imagenet --data-root {cache_dir} --method both"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
