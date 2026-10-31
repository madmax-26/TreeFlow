"""
Assertion script for ImageRF (run manually):

    python test_rgb_rf.py            # all sections
    python test_rgb_rf.py --quick    # skip the slow real-data sections (9-13)

Sections:
  1-8   patch/integral-image and node-test correctness (fast, synthetic)
  9-11  forest behaviour on real AID data (slow)
  12-13 TreeFlow round-trip and whole-image prediction (slow)
"""

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rgb_patch import (FEATURE_CHANNELS, N_FEATURE_CHANNELS, PreparedPatch, as_prepared,
                       derived_channels, integral_image, to_uint8_rgb)
from rgb_node_tests import OPERATORS, PROJECTIONS, RectTest, RectTestSampler

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(ok: bool, msg: str) -> None:
    global failures
    print(f"  {PASS if ok else FAIL}  {msg}")
    if not ok:
        failures += 1


def brute_mean(arr, r, c, rh, rw, ch) -> float:
    """Reference implementation the SAT must reproduce exactly."""
    return float(arr[r:r + rh, c:c + rw, ch].astype(np.int64).sum()) / (rh * rw)


# ── 1. Integral rectangle means ──────────────────────────────────────────────

def test_rect_means(rng):
    print("Integral rectangle means:")
    S = 64
    patches = {
        "random":    rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8),
        "gradient":  np.tile(np.linspace(0, 255, S, dtype=np.uint8)[None, :, None], (S, 1, 3)),
        "constant":  np.full((S, S, 3), 77, dtype=np.uint8),
        "saturated": np.full((S, S, 3), 255, dtype=np.uint8),
    }
    worst_rel = 0.0
    exact_ok = True
    for name, arr in patches.items():
        p = PreparedPatch(arr)
        for _ in range(500):
            rh = int(rng.integers(1, S + 1))
            rw = int(rng.integers(1, S + 1))
            r = int(rng.integers(0, S - rh + 1))
            c = int(rng.integers(0, S - rw + 1))
            ch = int(rng.integers(0, 3))
            got = p.rect_mean(r, c, rh, rw, ch)
            want = brute_mean(arr, r, c, rh, rw, ch)
            if got != want:
                exact_ok = False
            ref = float(arr[r:r + rh, c:c + rw, ch].mean())
            if ref != 0:
                worst_rel = max(worst_rel, abs(got - ref) / abs(ref))
    check(exact_ok, "rect_mean is EXACTLY the brute-force integer mean (2000 rects, 4 patch types)")
    check(worst_rel < 1e-12, f"agrees with numpy .mean() to rtol {worst_rel:.1e}")

    # int32 headroom: the largest possible sum must not overflow
    full = np.full((S, S, 3), 255, dtype=np.uint8)
    tab = integral_image(full)
    check(tab.dtype == np.int32 and int(tab[-1, -1, 0]) == 255 * S * S,
          f"int32 SAT holds the max sum ({255 * S * S})")


# ── 2. Edge and degenerate rectangles ────────────────────────────────────────

def test_edge_rects(rng):
    print("Edge / degenerate rectangles:")
    S = 16
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    p = PreparedPatch(arr)
    cases = [
        ("top-left corner",  0, 0, 1, 1),
        ("bottom-right",     S - 1, S - 1, 1, 1),
        ("full patch",       0, 0, S, S),
        ("full row strip",   0, 0, 1, S),
        ("full col strip",   0, 0, S, 1),
        ("touching right",   3, S - 4, 2, 4),
        ("touching bottom",  S - 4, 3, 4, 2),
    ]
    ok = True
    for name, r, c, rh, rw in cases:
        for ch in range(3):
            if p.rect_mean(r, c, rh, rw, ch) != brute_mean(arr, r, c, rh, rw, ch):
                ok = False
                check(False, f"{name} mismatch")
    check(ok, f"all {len(cases)} edge/degenerate rectangle shapes exact")


# ── 3. center operator ───────────────────────────────────────────────────────

def test_center(rng):
    print("center operator:")
    S = 32
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    p = PreparedPatch(arr)
    ok_sem, ok_1x1 = True, True
    for _ in range(2000):
        rh = int(rng.integers(1, S + 1))
        rw = int(rng.integers(1, S + 1))
        r = int(rng.integers(0, S - rh + 1))
        c = int(rng.integers(0, S - rw + 1))
        ch = int(rng.integers(0, 3))
        if p.rect_center(r, c, rh, rw, ch) != arr[r + rh // 2, c + rw // 2, ch]:
            ok_sem = False
    check(ok_sem, "rect_center == arr[r + rh//2, c + rw//2, ch] (2000 rects)")

    for r in range(S):
        for c in range(S):
            for ch in range(3):
                if p.rect_center(r, c, 1, 1, ch) != p.rect_mean(r, c, 1, 1, ch):
                    ok_1x1 = False
    check(ok_1x1, "center == mean exactly on every 1x1 rect (the pixel-pair test)")


# ── 4. Projection determinism ────────────────────────────────────────────────

def test_determinism(rng):
    print("Projection determinism:")
    S = 64
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    prep = PreparedPatch(arr)
    sampler = RectTestSampler(patch_size=S)
    tests = [sampler.sample(rng) for _ in range(200)]

    repeat_ok = all(len({t.project(prep) for _ in range(100)}) == 1 for t in tests[:20])
    check(repeat_ok, "same test x same patch x 100 calls -> one identical value")

    raw_ok = all(t.project(prep) == t.project(arr) for t in tests)
    check(raw_ok, "prepared and raw input give identical values (one code path)")

    rt_ok = True
    for t in tests:
        clone = RectTest(**dataclasses.asdict(t))
        if clone.project(prep) != t.project(prep):
            rt_ok = False
    check(rt_ok, "dataclass round-trip preserves the projection exactly")

    # every projection type must actually be exercised and finite
    seen = {t.projection for t in tests}
    check(seen == set(PROJECTIONS), f"all projection types sampled: {sorted(seen)}")
    check(all(np.isfinite(t.project(prep)) for t in tests), "all projections finite")


# ── 5. The "1pt" guard stays inert ───────────────────────────────────────────

def test_guard(rng):
    print('PolSAR "1pt" guard stays inert:')
    check("1pt" not in PROJECTIONS, '"1pt" is not a PROJECTIONS entry')
    sampler = RectTestSampler(patch_size=32)
    tests = [sampler.sample(rng) for _ in range(5000)]
    check(all(t.projection != "1pt" for t in tests), "no sampled test has projection == '1pt'")
    check(all(t.reference is None for t in tests), "every sampled test has reference is None")

    # The real thing: the reused core must keep every candidate it draws.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PolSAR"))
    from polsar_online_rf import OnlinePolSARDecisionTree
    tree = OnlinePolSARDecisionTree(sampler=sampler, n_candidates=64, n_classes=3, random_state=0)
    cands = tree._sample_candidates()
    check(len(cands) == 64, f"_sample_candidates kept all 64 candidates (got {len(cands)})")

    try:
        RectTestSampler(patch_size=32, projections=["1pt"])
        check(False, "sampler rejects a '1pt' projection")
    except ValueError:
        check(True, "sampler rejects a '1pt' projection")


# ── 6. Duck-typed contract ───────────────────────────────────────────────────

def test_contract():
    print("Duck-typed NodeTest / sampler contract:")
    t = RectTest(projection="diff", rects=[(0, 0, 2, 2), (2, 2, 2, 2)],
                 operators=["mean", "mean"], channels=[0, 0])
    for attr in ("project", "evaluate", "threshold", "projection", "reference"):
        check(hasattr(t, attr), f"RectTest has .{attr}")
    t.threshold = 1.5
    check(t.threshold == 1.5, "RectTest.threshold is writable (set by _fix_split)")
    s = RectTestSampler(patch_size=16)
    check(hasattr(s, "sample") and hasattr(s, "patch_size"),
          "RectTestSampler has .sample and .patch_size")
    check(isinstance(t.evaluate(PreparedPatch(np.zeros((16, 16, 3), np.uint8))), bool),
          "evaluate returns a bool")


# ── 7. Log-uniform rectangle sizing ──────────────────────────────────────────

def test_sizing(rng):
    print("Log-uniform rectangle sizing:")
    S = 64
    sampler = RectTestSampler(patch_size=S, min_rect_size=1, max_rect_size=S)
    hs, ws, rs, cs = [], [], [], []
    for _ in range(100_000):
        r, c, rh, rw = sampler._sample_rect(rng)
        hs.append(rh); ws.append(rw); rs.append(r); cs.append(c)
    hs = np.array(hs); ws = np.array(ws); rs = np.array(rs); cs = np.array(cs)

    check(hs.min() == 1 and hs.max() == S, f"heights span the full range [{hs.min()}, {hs.max()}]")
    check(ws.min() == 1 and ws.max() == S, f"widths span the full range [{ws.min()}, {ws.max()}]")

    octaves = [((hs >= 2 ** k) & (hs < 2 ** (k + 1))).sum() for k in range(6)]
    lo, hi = min(octaves), max(octaves)
    flat = (hi - lo) / np.mean(octaves) < 0.30
    check(flat, f"per-octave counts are flat within tolerance: {octaves}")

    corr = float(np.corrcoef(hs, ws)[0, 1])
    check(abs(corr) < 0.05, f"height and width are independent (corr {corr:+.4f})")
    check(bool(np.all(rs + hs <= S)) and bool(np.all(cs + ws <= S)),
          "every rectangle lies inside the patch")

    # positions should cover the patch, not cluster at the centre like PolSAR's
    check(rs.min() == 0 and rs.max() == S - 1, "row offsets reach both edges (scene mode)")


# ── 8. Derived channels and the std operator ─────────────────────────────────

def test_features(rng):
    print("Derived channels and std operator:")
    from scipy import ndimage

    S = 64
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    arr[0, 0] = 0                                    # one black pixel
    d = derived_channels(arr)
    f = arr.astype(np.float64)
    tot = f.sum(axis=2)
    nz = tot > 0
    g_ref = np.where(nz, 255 * f[:, :, 1] / np.where(nz, tot, 1), 85.0)
    b_ref = np.where(nz, 255 * f[:, :, 2] / np.where(nz, tot, 1), 85.0)
    check(np.array_equal(d[:, :, 0], np.clip(np.rint(g_ref), 0, 255).astype(np.uint8)),
          "g_chroma == 255 * G / (R+G+B)")
    check(np.array_equal(d[:, :, 1], np.clip(np.rint(b_ref), 0, 255).astype(np.uint8)),
          "b_chroma == 255 * B / (R+G+B)")
    check(d[0, 0, 0] == 85 and d[0, 0, 1] == 85, "a black pixel gets neutral chromaticity 85")

    # Chromaticity must ignore brightness: halving a patch keeps it (up to rounding).
    bright = rng.integers(40, 256, size=(S, S, 3), dtype=np.uint8)
    dim = (bright // 2 * 2).astype(np.uint8)
    half = (dim // 2).astype(np.uint8)
    diff = np.abs(derived_channels(dim)[:, :, :2].astype(int)
                  - derived_channels(half)[:, :, :2].astype(int))
    check(diff.max() <= 1, f"chromaticity is brightness-invariant (max diff {diff.max()})")

    lum = tot / 3.0
    mag = np.hypot(ndimage.sobel(lum, axis=1, mode="nearest"),
                   ndimage.sobel(lum, axis=0, mode="nearest")) / 4.0
    check(np.array_equal(d[:, :, 2], np.clip(np.rint(mag), 0, 255).astype(np.uint8)),
          "gradient == Sobel magnitude / 4 (matches scipy.ndimage)")
    flat = derived_channels(np.full((S, S, 3), 120, np.uint8))
    check(flat[:, :, 2].max() == 0, "a flat patch has zero gradient")
    step = np.zeros((S, S, 3), np.uint8)
    step[:, S // 2:] = 255
    check(derived_channels(step)[:, :, 2].max() == 255, "a full 0->255 step reaches 255")

    p = PreparedPatch(arr)
    feats = np.concatenate([arr, d], axis=2)
    ok_mean = ok_std = ok_ctr = True
    worst = 0.0
    for _ in range(2000):
        rh = int(rng.integers(1, S + 1)); rw = int(rng.integers(1, S + 1))
        r = int(rng.integers(0, S - rh + 1)); c = int(rng.integers(0, S - rw + 1))
        ch = int(rng.integers(0, N_FEATURE_CHANNELS))
        region = feats[r:r + rh, c:c + rw, ch].astype(np.float64)
        if p.rect_mean(r, c, rh, rw, ch) != brute_mean(feats, r, c, rh, rw, ch):
            ok_mean = False
        got, want = p.rect_std(r, c, rh, rw, ch), float(region.std())
        worst = max(worst, abs(got - want))
        if got < 0 or not np.isfinite(got):
            ok_std = False
        if p.rect_center(r, c, rh, rw, ch) != feats[r + rh // 2, c + rw // 2, ch]:
            ok_ctr = False
    check(ok_mean, "rect_mean is exact on all 6 feature channels")
    check(ok_std and worst < 1e-9, f"rect_std == numpy std (2000 rects, max abs err {worst:.1e})")
    check(ok_ctr, "rect_center reads derived channels too")
    check(all(p.rect_std(r, c, 1, 1, ch) == 0 for r in range(0, S, 7)
              for c in range(0, S, 7) for ch in range(N_FEATURE_CHANNELS)),
          "std of a 1x1 rectangle is exactly 0")
    check(PreparedPatch(np.full((S, S, 3), 200, np.uint8)).rect_std(0, 0, S, S, 0) == 0,
          "std of a constant patch is exactly 0")

    # Squared tables are int64: a large saturated patch must not overflow.
    big = np.zeros((256, 256, 3), np.uint8)
    big[::2] = 255
    pb = PreparedPatch(big)
    check(abs(pb.rect_std(0, 0, 256, 256, 0) - 127.5) < 1e-9,
          "std stays exact on a 256x256 patch (int64 squared tables)")

    sampler = RectTestSampler(patch_size=S, cross_channel=True)
    tests = [sampler.sample(rng) for _ in range(3000)]
    check({o for t in tests for o in t.operators} == set(OPERATORS),
          f"every operator sampled: {OPERATORS}")
    check({c for t in tests for c in t.channels} == set(range(N_FEATURE_CHANNELS)),
          f"every feature channel sampled: {FEATURE_CHANNELS}")
    check(all(np.isfinite(t.project(p)) for t in tests), "all new-family projections finite")
    rgb_only = RectTestSampler(patch_size=S, channels=[0, 1, 2], operators=["mean", "center"])
    t2 = [rgb_only.sample(rng) for _ in range(500)]
    check(all(set(t.channels) <= {0, 1, 2} and "std" not in t.operators for t in t2),
          "the original colour-only family is still selectable")


def main() -> None:
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(7)

    test_rect_means(rng)
    test_edge_rects(rng)
    test_center(rng)
    test_determinism(rng)
    test_guard(rng)
    test_contract()
    test_sizing(rng)
    test_features(rng)

    if not quick:
        from test_rgb_forest import run_forest_sections
        run_forest_sections(rng)
        from test_rgb_treeflow import run_treeflow_section
        run_treeflow_section(rng)

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
