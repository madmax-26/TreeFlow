"""
End-to-end demo: train an online forest on EuroSAT (RGB) and watch it learn in
the TreeFlow frontend.

Run the backend:
    python example_eurosat_treeflow.py             # serves on http://localhost:8000

Then start the frontend:
    cd ../Frontend && npm run dev         # http://localhost:5173

Same shape as example_rgb_treeflow.py and example_fbp_treeflow.py. EuroSAT is
the dataset for EVALUATING the similarity views: its 10 classes are separable by
colour and texture, so the forest gets strong enough for its overlap matrices to
carry structure, and EUROSAT_GROUPS gives a semantic grouping to hold that
structure against. After training, the script prints that comparison:

  * within- vs between-group mean overlap, for leaf co-occurrence and path
    overlap — a forest that has learned something meaningful overlaps more
    inside a group (crops, vegetation, built, water) than across;

  * each class's most-overlapping neighbour, with the pairs that cross a group
    boundary marked — those are the confusions driven by appearance rather
    than meaning (River / Highway is the expected one).

Startup is quick: tiles are 64 px JPEGs, so the eval set builds in seconds.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eurosat_datasets import (eurosat_class_colors, eurosat_group_of,  # noqa: E402
                              eurosat_label_map, load_eurosat_dataset,
                              prepare_patch_stream, sample_patch_set,
                              train_test_split_scenes)
from rgb_node_tests import RectTestSampler  # noqa: E402
from rgb_forest import ImageOnlineRandomForest  # noqa: E402
from rgb_interface import (convert_image_rf, evaluate_image_rf,  # noqa: E402
                           make_image_router, patch_to_rgb)
from treeflow import TreeFlow, _compute_similarity  # noqa: E402  (Library path is added by rgb_interface)

# None = all 2000-3000 images per class. Set to e.g. 300 for a quick first run.
MAX_PER_CLASS = None

# $TREEFLOW_PORT overrides, for when a notebook kernel still holds 8000.
PORT = int(os.environ.get("TREEFLOW_PORT", 8002))
WORK_DIR = "vis-eurosat/"

# Split-test search. CROSS_CHANNEL=True lets a two-rectangle test compare
# different channels ("red here vs green there"), i.e. colour-opponency tests;
# N_CANDIDATES is how many random tests a node tries before it splits.
CROSS_CHANNEL = True
N_CANDIDATES = 100


def print_similarity_report(forest, class_names) -> None:
    """Hold the forest's overlap matrices against EUROSAT_GROUPS (module docstring)."""
    sim = _compute_similarity(forest)
    groups = eurosat_group_of(class_names)
    labels = eurosat_label_map(class_names)
    C = len(class_names)
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    print("\nSimilarity vs semantic groups:")
    for metric, M in sim["metrics"].items():
        within = [M[i][j] for i, j in pairs if groups[i] == groups[j]]
        between = [M[i][j] for i, j in pairs if groups[i] != groups[j]]
        print(f"  {metric:18} within {np.mean(within):.3f} | between {np.mean(between):.3f} "
              f"| ratio {np.mean(within) / max(np.mean(between), 1e-9):.2f}")

    M = sim["metrics"]["path_overlap"]
    print("  Nearest neighbour by path overlap (* = crosses a group boundary):")
    for i in range(C):
        j = max((k for k in range(C) if k != i), key=lambda k: M[i][k])
        mark = "*" if groups[i] != groups[j] else " "
        print(f"   {mark} {labels[i]:22} -> {labels[j]:22} {M[i][j]:.3f}")


def main():
    rng = np.random.default_rng(0)

    # ---- 1. Data ------------------------------------------------------------
    patch_size = 64
    per_image = 1
    batch_size = 2000
    eval_n = 1500
    epochs = 2

    print("Loading EuroSAT ...")
    ds = load_eurosat_dataset(max_per_class=MAX_PER_CLASS, rng=rng)
    train_ds, test_ds = train_test_split_scenes(ds, test_frac=0.2, rng=rng)
    print(f"  {len(ds)} tiles, {ds.n_classes} classes "
          f"({len(train_ds)} train / {len(test_ds)} test images)")

    print(f"Preparing {eval_n} evaluation patches ...")
    eval_patches, eval_labels = sample_patch_set(test_ds, n=eval_n, patch_size=patch_size, rng=rng)

    # ---- 2. Forest ----------------------------------------------------------
    sampler = RectTestSampler(patch_size=patch_size, min_rect_size=1,
                              max_rect_size=patch_size, cross_channel=CROSS_CHANNEL)
    rf = ImageOnlineRandomForest(
        sampler=sampler,
        n_trees=20,
        n_candidates=N_CANDIDATES,
        max_depth=24,
        tau=1000,
        tau_decay=0.7,
        tau_min=10,
        n_classes=ds.n_classes,
        random_state=42,
    )

    # ---- 3. TreeFlow -------------------------------------------------------------
    treeflow = TreeFlow(port=PORT, work_dir=WORK_DIR)
    treeflow.set_reference_router(make_image_router(rf))

    refs_per_class = 60
    for c in range(ds.n_classes):
        hits = np.where(eval_labels == c)[0]
        for i in range(min(refs_per_class, len(hits))):
            p = eval_patches[int(hits[i])]
            treeflow.add_reference(p, image=patch_to_rgb(p), label=ds.class_names[c],
                                   label_index=c, step=0,
                                   name=f"{ds.class_names[c]}-{i + 1}")

    label_map = eurosat_label_map(ds.class_names)
    class_colors = eurosat_class_colors(ds.class_names)

    # ---- 4. Learn batch by batch, push one TreeFlow iteration per batch ----------
    print(f"Training: {epochs} epoch(s) x {len(train_ds)} tiles x {per_image} subwindow(s) ...")
    t0 = time.perf_counter()
    forest = ev = None
    for patches, labels in prepare_patch_stream(train_ds, patch_size=patch_size,
                                                per_image=per_image,
                                                batch_size=batch_size,
                                                epochs=epochs, rng=rng):
        rf.partial_fit(patches, labels)

        ev = evaluate_image_rf(rf, eval_patches, eval_labels)
        forest = convert_image_rf(rf, class_labels=ds.class_names,
                                  tree_metrics=ev["tree_metrics"],
                                  forest_metrics=ev["forest_metrics"])
        forest.class_colors = class_colors
        idx = treeflow.push(forest, label_map=label_map)
        tree_acc = sum(m["accuracy"] for m in ev["tree_metrics"]) / len(ev["tree_metrics"])
        print(f"Pushed iteration {idx} — forest {ev['forest_accuracy']:.2%} "
              f"vs ⌀ single tree {tree_acc:.2%} "
              f"(chance {1 / ds.n_classes:.2%}, {time.perf_counter() - t0:.0f}s elapsed)")

    if forest is not None:
        print_similarity_report(forest, ds.class_names)

    print(f"\nBackend running at http://localhost:{PORT}")
    print("Start the frontend with:  cd ../Frontend && npm run dev")
    print("Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down ...")
        treeflow.shutdown()


if __name__ == "__main__":
    main()
