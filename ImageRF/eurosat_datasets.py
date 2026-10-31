"""
EuroSAT (RGB) scene-classification dataset.

    Helber, Bischke, Dengel & Borth, "EuroSAT: A Novel Dataset and Deep Learning
    Benchmark for Land Use and Land Cover Classification", IEEE JSTARS 2019

27,000 Sentinel-2 patches, 64 x 64 px at 10 m GSD, 10 classes, one label per
patch. It is a SCENE dataset like AID/UC-Merced, so it reuses rgb_datasets'
SceneDataset, split and streaming code unchanged; only two things differ:

  * The class is the FOLDER name (<root>/<Class>/<Class>_<n>.jpg), not a token
    parsed out of a flat directory. The filename prefix is still checked
    against the folder, so a file dropped into the wrong class folder is
    reported instead of silently mislabelled.

  * The tiles are already 64 px — the forest's patch size. AID's 600 px tiles
    are DOWNscaled 5-10x by Marée subwindows; here any subwindow is UPscaled.
    The default size range is therefore narrowed to 75-100% (the wrappers
    below), so a patch is never blown up more than ~1.3x and stays sharp
    enough to read in NodeView.

Why this dataset exists in the repo: its classes are separable by colour and
texture — the only thing rectangle tests see — so the forest reaches a useful
accuracy, while the classes still fall into semantic groups (EUROSAT_GROUPS)
that the similarity metrics can be checked against. That is the evaluation
Five-Billion-Pixels cannot support: there most of the 24 classes differ by land
USE, the forest stays weak, and the overlap matrices are uniformly blurred.

Layout expected under `root` (as extracted from Zenodo record 7711810,
EuroSAT_RGB.zip):

    AnnualCrop/AnnualCrop_1.jpg ... SeaLake/SeaLake_3000.jpg
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Optional

import numpy as np

import rgb_datasets
from rgb_datasets import SceneDataset, train_test_split_scenes  # noqa: F401  (re-exported)

# Dataset location: $EUROSAT_ROOT if set, else <repo>/big_datasets/EuroSAT/EuroSAT_RGB.
EUROSAT_ROOT = os.environ.get("EUROSAT_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "big_datasets", "EuroSAT", "EuroSAT_RGB")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

# Folder names, sorted — the same order load_eurosat_dataset assigns indexes in.
EUROSAT_CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
N_EUROSAT_CLASSES = len(EUROSAT_CLASS_NAMES)    # 10

# Semantic grouping used as GROUND TRUTH for the similarity evaluation: a forest
# that has learned something meaningful should overlap more within a group than
# across groups. Highway sits with the built classes semantically, but shares
# River's geometry (a thin line across the patch) — the pair the path overlap
# is most expected to expose as a non-semantic confusion.
EUROSAT_GROUPS = {
    "crops":      ["AnnualCrop", "PermanentCrop"],
    "vegetation": ["Forest", "HerbaceousVegetation", "Pasture"],
    "built":      ["Highway", "Industrial", "Residential"],
    "water":      ["River", "SeaLake"],
}

# One hue family per group, so a node's class mix in the frontend reads as
# "mostly vegetation" at a glance before the individual classes are told apart.
_CLASS_COLORS = {
    "AnnualCrop":           "#e6a817",
    "PermanentCrop":        "#b5651d",
    "Forest":               "#1b7837",
    "HerbaceousVegetation": "#7fbf3f",
    "Pasture":              "#c2e07a",
    "Highway":              "#7a7a7a",
    "Industrial":           "#c0392b",
    "Residential":          "#e377c2",
    "River":                "#3a8fd9",
    "SeaLake":              "#0b3d91",
}

_FILE_RE = re.compile(r"^(?P<cls>[A-Za-z]+)_\d+\.(jpg|jpeg|png|tif|tiff)$", re.IGNORECASE)


def eurosat_label_map(class_names: Optional[list] = None) -> dict:
    """{0: 'Annual Crop', 1: 'Forest', ...} for treeflow.push(label_map=...)."""
    names = class_names or EUROSAT_CLASS_NAMES
    return {i: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n) for i, n in enumerate(names)}


def eurosat_class_colors(class_names: Optional[list] = None) -> list:
    """Group-coloured hex palette for Forest.class_colors (see _CLASS_COLORS)."""
    return [_CLASS_COLORS[n] for n in (class_names or EUROSAT_CLASS_NAMES)]


def eurosat_group_of(class_names: Optional[list] = None) -> list:
    """Group name of each class index, e.g. for scoring within- vs between-group
    overlap of a similarity matrix."""
    lookup = {c: g for g, cs in EUROSAT_GROUPS.items() for c in cs}
    return [lookup[n] for n in (class_names or EUROSAT_CLASS_NAMES)]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_eurosat_dataset(
    root: str = EUROSAT_ROOT,
    max_per_class: Optional[int] = None,
    rng=None,
) -> SceneDataset:
    """Scan the class folders into a SceneDataset.

    max_per_class: keep at most this many images per class — EuroSAT has
                   2000-3000 per class, so this is the knob for a quick run.

    Files whose name does not start with their folder's class are skipped with
    a warning rather than trusted, as are unknown folders.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset directory not found: {root}")

    by_class: dict[str, list] = {}
    skipped = 0
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        if cls not in EUROSAT_CLASS_NAMES:
            warnings.warn(f"unknown class folder {cls!r} in {root} was skipped", stacklevel=2)
            continue
        for fname in sorted(os.listdir(cdir)):
            m = _FILE_RE.match(fname)
            if not m:
                continue
            if m.group("cls") != cls:
                skipped += 1
                continue
            by_class.setdefault(cls, []).append(os.path.join(cdir, fname))
    if skipped:
        warnings.warn(f"{skipped} file(s) under {root} sit in a folder that does not "
                      "match their filename's class and were skipped", stacklevel=2)
    if not by_class:
        raise ValueError(f"no EuroSAT class folders found in {root}")

    rng = rng or np.random.default_rng(0)
    # Index by the FULL class list, so a partial download keeps the same label
    # numbering (and colours) as the complete dataset.
    paths, labels = [], []
    for idx, cls in enumerate(EUROSAT_CLASS_NAMES):
        files = by_class.get(cls, [])
        if max_per_class is not None and len(files) > max_per_class:
            pick = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(pick)]
        paths.extend(files)
        labels.extend([idx] * len(files))

    missing = [c for c in EUROSAT_CLASS_NAMES if c not in by_class]
    if missing:
        warnings.warn(f"class folder(s) missing from {root}: {', '.join(missing)}", stacklevel=2)

    return SceneDataset(name="EUROSAT", paths=paths, labels=np.array(labels, dtype=int),
                        class_names=list(EUROSAT_CLASS_NAMES))


# ---------------------------------------------------------------------------
# Sampling — rgb_datasets with EuroSAT's subwindow range (module docstring)
# ---------------------------------------------------------------------------

EUROSAT_SUBWINDOW = dict(min_frac=0.75, max_frac=1.0)


def sample_patch_set(ds: SceneDataset, n: int, patch_size: int = 64, rng=None,
                     prepared: bool = True, **subwindow_kw):
    """rgb_datasets.sample_patch_set with EUROSAT_SUBWINDOW as the default range."""
    return rgb_datasets.sample_patch_set(ds, n, patch_size=patch_size, rng=rng,
                                         prepared=prepared,
                                         **{**EUROSAT_SUBWINDOW, **subwindow_kw})


def prepare_patch_stream(ds: SceneDataset, patch_size: int = 64, per_image: int = 1,
                         batch_size: int = 2000, epochs: int = 1, rng=None,
                         **subwindow_kw):
    """rgb_datasets.prepare_patch_stream with EUROSAT_SUBWINDOW as the default range.

    per_image defaults to 1 rather than AID's 4: a 64 px tile holds little more
    than one 75-100% subwindow's worth of distinct content, so extra draws from
    the same tile add near-duplicates rather than information. Use epochs for
    a longer stream instead.
    """
    return rgb_datasets.prepare_patch_stream(ds, patch_size=patch_size, per_image=per_image,
                                             batch_size=batch_size, epochs=epochs, rng=rng,
                                             **{**EUROSAT_SUBWINDOW, **subwindow_kw})
