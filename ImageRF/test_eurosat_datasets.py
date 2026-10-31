"""
Assertion suite for eurosat_datasets.

Runs standalone:  python test_eurosat_datasets.py

Sections 1-3 build a synthetic class-folder tree, so they run anywhere.
Section 4 only runs if the extracted dataset is present, and checks what
CANNOT be checked synthetically: the real folder names, counts and tile size.
"""

import os
import sys
import tempfile
import warnings

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eurosat_datasets as E  # noqa: E402
from rgb_patch import PreparedPatch  # noqa: E402

_failures = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        _failures.append(label)


def write_tile(path, value):
    Image.fromarray(np.full((64, 64, 3), value, np.uint8)).save(path)


# ── 1. Class table ───────────────────────────────────────────────────────────
print("\n1. class table")
check(E.N_EUROSAT_CLASSES == 10, "10 classes")
check(E.EUROSAT_CLASS_NAMES == sorted(E.EUROSAT_CLASS_NAMES), "class list is sorted")
grouped = [c for cs in E.EUROSAT_GROUPS.values() for c in cs]
check(sorted(grouped) == E.EUROSAT_CLASS_NAMES, "groups cover every class exactly once")
groups = E.eurosat_group_of()
check(groups[E.EUROSAT_CLASS_NAMES.index("River")] == "water", "River is in the water group")
colors = E.eurosat_class_colors()
check(len(colors) == 10 and len(set(colors)) == 10, "10 distinct hex colours")
check(all(c.startswith("#") and len(c) == 7 for c in colors), "hex format")
lm = E.eurosat_label_map()
check(lm[0] == "Annual Crop" and lm[2] == "Herbaceous Vegetation" and lm[9] == "Sea Lake",
      "label map splits CamelCase")

# ── 2. Loader on a synthetic tree ────────────────────────────────────────────
print("\n2. loader")
with tempfile.TemporaryDirectory() as root:
    for cls, n in [("Forest", 5), ("River", 3), ("SeaLake", 4)]:
        os.makedirs(os.path.join(root, cls))
        for i in range(1, n + 1):
            write_tile(os.path.join(root, cls, f"{cls}_{i}.jpg"), 40 * i)
    # A misfiled tile must be rejected, not given its folder's label.
    write_tile(os.path.join(root, "Forest", "River_99.jpg"), 0)
    # Non-image clutter is ignored without a warning.
    open(os.path.join(root, "Forest", "readme.txt"), "w").close()
    os.makedirs(os.path.join(root, "NotAClass"))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = E.load_eurosat_dataset(root)
    msgs = " | ".join(str(x.message) for x in w)
    check(len(ds) == 12, "12 correctly filed tiles loaded")
    check("does not match their filename's class" in msgs, "misfiled tile warned about")
    check("NotAClass" in msgs, "unknown folder warned about")
    check("missing" in msgs, "missing classes warned about")
    check(not any("River_99" in p for p in ds.paths), "misfiled tile skipped")
    # Partial downloads must keep the full dataset's indexes.
    by = {E.EUROSAT_CLASS_NAMES[l] for l in ds.labels}
    check(by == {"Forest", "River", "SeaLake"}, "labels name the right folders")
    check(set(ds.labels) == {1, 8, 9}, "indexes match the full class list")
    check(ds.n_classes == 10, "class_names is always the full list")
    check(all(os.path.basename(os.path.dirname(p)) == E.EUROSAT_CLASS_NAMES[l]
              for p, l in zip(ds.paths, ds.labels)), "every path sits in its label's folder")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capped = E.load_eurosat_dataset(root, max_per_class=2)
    check(len(capped) == 6, "max_per_class caps each class")

    tr, te = E.train_test_split_scenes(ds, test_frac=0.25, rng=np.random.default_rng(0))
    check(not (set(tr.paths) & set(te.paths)), "train/test images disjoint")
    check(len(tr) + len(te) == len(ds), "split covers every image")

    # ── 3. Sampling ──────────────────────────────────────────────────────────
    print("\n3. sampling")
    ps, ls = E.sample_patch_set(ds, 20, patch_size=64, rng=np.random.default_rng(0))
    check(len(ps) == len(ls) == 20, "sample_patch_set returns n patches")
    check(all(isinstance(p, PreparedPatch) for p in ps), "patches are prepared")
    check(all(p.arr.shape == (64, 64, 3) for p in ps), "patches are 64 x 64 x 3")
    batches = list(E.prepare_patch_stream(ds, batch_size=5, epochs=2,
                                          rng=np.random.default_rng(0)))
    check(sum(len(b[0]) for b in batches) == 2 * len(ds),
          "stream draws one patch per image per epoch by default")
    check(all(len(p) == len(l) for p, l in batches), "batches pair patches and labels")
    # Uniform tiles survive the resize unchanged, so a patch's value identifies
    # its source tile — and therefore its folder.
    first = {E.EUROSAT_CLASS_NAMES[l] for l in ls}
    check(first <= {"Forest", "River", "SeaLake"}, "sampled labels come from the tree")

    # The narrowed range must be the default but stay overridable.
    import rgb_datasets
    seen = {}
    orig = rgb_datasets.sample_subwindows

    def spy(img, n, patch_size=64, **kw):
        seen.update(kw)
        return orig(img, n, patch_size, **kw)

    rgb_datasets.sample_subwindows = spy
    try:
        E.sample_patch_set(ds, 2, rng=np.random.default_rng(0))
        check(seen.get("min_frac") == 0.75, "default min_frac is 0.75")
        E.sample_patch_set(ds, 2, rng=np.random.default_rng(0), min_frac=0.5)
        check(seen.get("min_frac") == 0.5, "min_frac can be overridden")
    finally:
        rgb_datasets.sample_subwindows = orig

# ── 4. Real files, if extracted ──────────────────────────────────────────────
print("\n4. real dataset")
if not os.path.isdir(E.EUROSAT_ROOT):
    print("  skipped — dataset not found at", E.EUROSAT_ROOT)
else:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        full = E.load_eurosat_dataset()
    check(not w, "real tree loads without warnings")
    check(len(full) == 27000, f"27000 images (got {len(full)})")
    counts = np.bincount(full.labels, minlength=10)
    check(counts.min() >= 2000 and counts.max() <= 3000, "2000-3000 images per class")
    check(all(c > 0 for c in counts), "every class present")

    from rgb_datasets import load_image
    rng = np.random.default_rng(0)
    for i in rng.choice(len(full), 5, replace=False):
        img = load_image(full.paths[i])
        check(img.shape == (64, 64, 3) and img.dtype == np.uint8,
              f"{os.path.basename(full.paths[i])} is 64 x 64 RGB uint8")
    check(img.std() > 2, "a real tile has visible contrast")

    tr, te = E.train_test_split_scenes(full, test_frac=0.2, rng=rng)
    check(not (set(tr.paths) & set(te.paths)), "real train/test images disjoint")
    check(abs(len(te) - 5400) <= 10, "20% held out")

print("\n" + "=" * 60)
if _failures:
    print(f"{len(_failures)} FAILURE(S):")
    for f in _failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
