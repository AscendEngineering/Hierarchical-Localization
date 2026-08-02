import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import h5py
import torch
from tqdm import tqdm

from . import logger, matchers
from .utils.base_model import dynamic_load, load_model
from .utils.cache import LRUCache
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    "superpoint+lightglue": {
        "output": "matches-superpoint-lightglue",
        "model": {
            "name": "lightglue",
            "features": "superpoint",
        },
    },
    "disk+lightglue": {
        "output": "matches-disk-lightglue",
        "model": {
            "name": "lightglue",
            "features": "disk",
        },
    },
    "aliked+lightglue": {
        "output": "matches-aliked-lightglue",
        "model": {
            "name": "lightglue",
            "features": "aliked",
        },
    },
    "superglue": {
        "output": "matches-superglue",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
        },
    },
    "superglue-fast": {
        "output": "matches-superglue-it5",
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 5,
        },
    },
    "NN-superpoint": {
        "output": "matches-NN-mutual-dist.7",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
        },
    },
    "NN-ratio": {
        "output": "matches-NN-mutual-ratio.8",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
        },
    },
    "NN-mutual": {
        "output": "matches-NN-mutual",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
        },
    },
    "adalam": {
        "output": "matches-adalam",
        "model": {"name": "adalam"},
    },
    # ONNX-accelerated LightGlue 
    "superpoint_onnx+lightglue_onnx": {
        "output": "matches-superpoint-lightglue-onnx",
        "model": {
            "name": "lightglue_onnx",
            "features": "superpoint",
        },
    },
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


@torch.no_grad()
def match_from_paths_fast(
    conf: Dict,
    pairs: List[Tuple[str, str]],
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    model=None,
    ref_cache: Optional[LRUCache] = None,
) -> None:
    """
    Optimized matching for inference: pre-loads features into memory to avoid
    repeated HDF5 I/O overhead. Much faster for single-query scenarios.
    
    Args:
        ref_cache: Optional LRU cache for reference features. If provided,
                   reference features are loaded through cache (for multi-query efficiency).
    """
    if len(pairs) == 0:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_model(matchers, conf, device)

    # Pre-load query features
    query_names = set(p[0] for p in pairs)
    ref_names = set(p[1] for p in pairs)
    
    query_features = {}
    with h5py.File(feature_path_q, "r") as fd:
        for name in query_names:
            grp = fd[name]
            query_features[name] = {
                k: torch.from_numpy(v.__array__()).float().to(device) 
                for k, v in grp.items() if k != "image_size"
            }
            query_features[name]["image_size"] = tuple(grp["image_size"])
    
    # If a reference cache is provided, use it to load reference features
    if ref_cache is not None:
        # Partition keys into hits and misses
        cache_hits, cache_misses = ref_cache.partition_keys(ref_names)
        ref_cache.record_stats(hits=len(cache_hits), misses=len(cache_misses))
        
        # Batch load all misses with single file open (much faster than per-miss opens)
        if cache_misses:
            with h5py.File(feature_path_ref, "r") as fd:
                for name in cache_misses:
                    grp = fd[name]
                    feats = {k: torch.from_numpy(v.__array__()).float()
                             for k, v in grp.items() if k != "image_size"}
                    feats["image_size"] = tuple(grp["image_size"])
                    ref_cache.put(name, feats)
        
        # Get all features from cache (now all should be present)
        ref_features = ref_cache.get_many(ref_names)
    # Else load directly from HDF5
    else:
        ref_features = {}
        with h5py.File(feature_path_ref, "r") as fd:
            for name in ref_names:
                grp = fd[name]
                ref_features[name] = {
                    k: torch.from_numpy(v.__array__()).float().to(device)
                    for k, v in grp.items() if k != "image_size"
                }
                ref_features[name]["image_size"] = tuple(grp["image_size"])

    # Match all pairs (pure GPU, no transfers in loop)
    match_path.parent.mkdir(exist_ok=True, parents=True)
    results = []
    
    for name0, name1 in tqdm(pairs, smoothing=0.1):
        # Build data dict from pre-loaded GPU features (no .to(device) needed)
        q_feats = query_features[name0]
        r_feats = ref_features[name1]
        
        data = {
            f"{k}0": v.unsqueeze(0) for k, v in q_feats.items() if k != "image_size"
        }
        data.update({
            f"{k}1": v.unsqueeze(0) for k, v in r_feats.items() if k != "image_size"
        })
        # Image tensors just for shape info (not used in ONNX matching)
        data["image0"] = {"shape": (1, 1) + q_feats["image_size"][::-1]}
        data["image1"] = {"shape": (1, 1) + r_feats["image_size"][::-1]}
        
        # Run matcher
        pred = model(data)
        pair_name = names_to_pair(name0, name1)
        results.append((pair_name, pred))

    # Write all results at once
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        for pair_name, pred in results:
            if pair_name in fd:
                del fd[pair_name]
            grp = fd.create_group(pair_name)
            matches = pred["matches0"][0].cpu().short().numpy()
            grp.create_dataset("matches0", data=matches)
            if "matching_scores0" in pred:
                scores = pred["matching_scores0"][0].cpu().half().numpy()
                grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
    model = None,
    ref_cache: Optional[LRUCache] = None,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite, model, ref_cache)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


def get_model(conf: Dict, device: str = None):
    """Pre-load matcher model for reuse across multiple match calls."""
    return load_model(matchers, conf, device)


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
    model = None,
    ref_cache: Optional[LRUCache] = None,
) -> Path:
    logger.info(
        "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return

    # Use fast path when model is pre-loaded (inference mode)
    # This pre-loads features into memory to avoid HDF5 I/O overhead
    if model is not None:
        match_from_paths_fast(conf, pairs, match_path, feature_path_q, feature_path_ref, model, ref_cache)
        logger.info("Finished exporting matches.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(matchers, conf, device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logger.info("Finished exporting matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="feats-superpoint-n4096-r1024")
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="superglue", choices=list(confs.keys())
    )
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
