from pathlib import Path
from typing import Mapping, Tuple

import cv2
import h5py
import numpy as np
import pycolmap

from .parsers import names_to_pair, names_to_pair_old


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(
    path: Path, name: str, return_uncertainty: bool = False
) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f"Could not find pair {(name0, name1)}... "
        "Maybe you matched with a different list of pairs? "
    )


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def get_matches_batch(
    path: Path, query_name: str, db_names: list
) -> dict:
    """
    Load all matches for a query against multiple db images in one HDF5 open.
    
    Avoids the overhead of opening/closing the file for each pair.
    
    Args:
        path: Path to matches HDF5 file
        query_name: Name of the query image
        db_names: List of database image names to load matches for
        
    Returns:
        Dict mapping db_name -> (matches, scores) tuples
        Missing pairs are silently skipped.
    """
    results = {}
    with h5py.File(str(path), "r", libver="latest") as hfile:
        # For every database image
        for db_name in db_names:
            try:
                # Try to find the pair in the HDF5 file
                pair, reverse = find_pair(hfile, query_name, db_name)
                matches = hfile[pair]["matches0"].__array__()
                scores = hfile[pair]["matching_scores0"].__array__()
                
                # Filter valid matches
                idx = np.where(matches != -1)[0]
                matches = np.stack([idx, matches[idx]], -1)
                
                # If the pair was stored in reverse order, flip the matches
                if reverse: matches = np.flip(matches, -1)
                scores = scores[idx]
                
                results[db_name] = (matches, scores)
            except ValueError:
                # Pair not found, skip
                continue
    return results


def write_poses(
    poses: Mapping[str, pycolmap.Rigid3d], path: str, prepend_camera_name: bool
):
    with open(path, "w") as f:
        for query, t in poses.items():
            qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]]))
            tvec = " ".join(map(str, t.translation))
            name = query.split("/")[-1]
            if prepend_camera_name:
                name = query.split("/")[-2] + "/" + name
            f.write(f"{name} {qvec} {tvec}\n")
