"""
RGB patch preparation: summed-area tables for O(1) rectangle aggregates.

A patch is an (h, w, 3) uint8 image. Every node test in rgb_node_tests.py is a
comparison of rectangle aggregates, so the whole forest rests on one primitive:
the mean of an arbitrary rectangle, in constant time.

That is what the summed-area table (integral image) gives:

    S    = I[r+rh, c+rw] - I[r, c+rw] - I[r+rh, c] + I[r, c]
    mean = S / (rh * rw)

four lookups regardless of rectangle size. The table is built ONCE per patch
(PreparedPatch) and reused across every tree, node and candidate of the forest
— the structural advantage over the PolSAR forest, where each projection
recomputes its region operators from scratch.

Two extensions ride on the same primitive:

  * DERIVED CHANNELS (FEATURE_CHANNELS 3-5) are computed once per patch next to
    R, G, B and get their own tables, so a test reads "greenness" or "edge
    density" exactly as cheaply as a colour. They are the cues a rectangle mean
    of a raw colour cannot express: green/blue CHROMATICITY is a ratio, and so
    invariant to illumination (a dark forest and a bright meadow are both
    green); gradient magnitude responds to structure (roads, rivers, building
    edges) rather than to colour.

  * A second table of SQUARED values gives the rectangle standard deviation —
    texture — from the same four lookups:  var = E[x^2] - E[x]^2.
"""

from __future__ import annotations

import array

import numpy as np

N_CHANNELS = 3          # displayed image channels: R, G, B

# Channels a node test can read. 0-2 are the image; 3-5 are derived per patch
# by derived_channels and stored as uint8 like the image, so every table and
# operator treats them identically.
FEATURE_CHANNELS = ["R", "G", "B", "g_chroma", "b_chroma", "gradient"]
N_FEATURE_CHANNELS = len(FEATURE_CHANNELS)


def integral_image(arr: np.ndarray) -> np.ndarray:
    """(h, w, C) uint8 -> (h+1, w+1, C) int32 summed-area table.

    The zero first row/column is what lets rectangles touching the top/left
    edge be read without any boundary special-casing. int32 is exact here:
    the largest possible sum is 255 * h * w (255*64*64 = 1_044_480 << 2**31).
    """
    h, w, c = arr.shape
    out = np.zeros((h + 1, w + 1, c), dtype=np.int32)
    np.cumsum(np.cumsum(arr, axis=0, dtype=np.int32), axis=1, out=out[1:, 1:])
    return out


def derived_channels(arr: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 RGB -> (h, w, 3) uint8 [g_chroma, b_chroma, gradient].

    g_chroma, b_chroma:  255 * G / (R+G+B) and 255 * B / (R+G+B). A black
                         pixel has no chromaticity and gets the neutral 85
                         (255 / 3) rather than a division by zero.
    gradient:            Sobel magnitude of the luminance (R+G+B)/3, divided by
                         4 and clipped. A full 0->255 step gives
                         4*255/4 = 255; the rare stronger diagonal responses
                         clip, which costs nothing a threshold test could use.
    """
    f = arr.astype(np.float64)
    total = f.sum(axis=2)
    safe = np.where(total > 0, total, 1.0)
    g = np.where(total > 0, 255.0 * f[:, :, 1] / safe, 255.0 / 3)
    b = np.where(total > 0, 255.0 * f[:, :, 2] / safe, 255.0 / 3)

    lum = np.pad(total / 3.0, 1, mode="edge")
    gx = ((lum[:-2, 2:] + 2 * lum[1:-1, 2:] + lum[2:, 2:])
          - (lum[:-2, :-2] + 2 * lum[1:-1, :-2] + lum[2:, :-2]))
    gy = ((lum[2:, :-2] + 2 * lum[2:, 1:-1] + lum[2:, 2:])
          - (lum[:-2, :-2] + 2 * lum[:-2, 1:-1] + lum[:-2, 2:]))
    grad = np.hypot(gx, gy) / 4.0

    out = np.stack([g, b, grad], axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class PreparedPatch:
    """An (h, w, 3) uint8 patch plus everything a RectTest needs to evaluate.

    Built once per patch and handed to the forest in place of the raw array;
    the reused PolSAR forest core passes patches through opaquely, so it never
    needs to know.

    Storage choices are measured, not guessed (see bench_rgb_rf.py):
      * the SAT is one flat array.array('i') per channel — ~3x faster per
        projection than scalar indexing into a 3-D ndarray, at the same memory.
      * pixels are one interleaved `bytes`, so the `center` operator gets a
        plain int without numpy scalar overhead.
      * the squared-value tables are array.array('q') (int64): their largest
        sum, 255^2 * h * w, overflows int32 above ~180x180 px.
    Budget is roughly 330 KB per prepared 64x64 patch (six channels, an int32
    and an int64 table each — measured by bench_rgb_rf.py), so batch size is a
    memory knob as much as a training one.
    """

    __slots__ = ("arr", "h", "w", "_sat", "_sat2", "_px", "_stride")

    def __init__(self, arr: np.ndarray):
        a = np.asarray(arr)
        if a.ndim == 2:                      # grayscale -> replicate to RGB
            a = np.repeat(a[:, :, None], N_CHANNELS, axis=2)
        if a.ndim != 3 or a.shape[2] != N_CHANNELS:
            raise ValueError(f"expected an (h, w, 3) RGB patch, got {np.asarray(arr).shape}")
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        # Copy: a patch is usually a view into a much larger tile, and keeping
        # the view alive would pin the whole tile in memory.
        a = np.ascontiguousarray(a)

        self.arr = a
        self.h, self.w = int(a.shape[0]), int(a.shape[1])
        self._stride = self.w + 1

        feats = np.concatenate([a, derived_channels(a)], axis=2)
        table = integral_image(feats)
        sq = feats.astype(np.int64) ** 2
        table2 = np.zeros((self.h + 1, self.w + 1, N_FEATURE_CHANNELS), dtype=np.int64)
        np.cumsum(np.cumsum(sq, axis=0), axis=1, out=table2[1:, 1:])
        self._sat = [
            array.array("i", np.ascontiguousarray(table[:, :, ch]).ravel().tolist())
            for ch in range(N_FEATURE_CHANNELS)
        ]
        self._sat2 = [
            array.array("q", np.ascontiguousarray(table2[:, :, ch]).ravel().tolist())
            for ch in range(N_FEATURE_CHANNELS)
        ]
        self._px = feats.tobytes()           # index: (r * w + c) * N_FEATURE_CHANNELS + ch

    # ── O(1) region operators ────────────────────────────────────────────────

    def rect_mean(self, r: int, c: int, rh: int, rw: int, ch: int) -> float:
        """Mean of arr[r:r+rh, c:c+rw, ch] — four SAT lookups, any size.

        Exact and deterministic: the table is integer, so this is an integer
        sum divided by an integer area.
        """
        f = self._sat[ch]
        s = self._stride
        top = r * s + c
        bot = (r + rh) * s + c
        return (f[bot + rw] - f[top + rw] - f[bot] + f[top]) / (rh * rw)

    def rect_center(self, r: int, c: int, rh: int, rw: int, ch: int) -> int:
        """Value of the rectangle's centre pixel.

        Deliberately does NOT use the SAT (it is a single sample, not a sum).
        The `rh // 2` convention mirrors PolSAR's apply_operator "center"
        (region[rw // 2, rw // 2]), and it makes `center` and `mean` agree
        exactly on a 1x1 rectangle — which is how the classic Lepetit-Fua
        pixel-pair test falls out of either operator choice.
        """
        return self._px[((r + rh // 2) * self.w + (c + rw // 2)) * N_FEATURE_CHANNELS + ch]

    def rect_std(self, r: int, c: int, rh: int, rw: int, ch: int) -> float:
        """Standard deviation of arr[r:r+rh, c:c+rw, ch] — eight SAT lookups.

        Texture: a calm water surface and a forest canopy can share a mean
        colour but not a spread. Computed in integers up to the final division
        (n * sum(x^2) - sum(x)^2 is exact), so it is never negative from
        rounding and a 1x1 rectangle is exactly 0.
        """
        s = self._stride
        top = r * s + c
        bot = (r + rh) * s + c
        f = self._sat[ch]
        f2 = self._sat2[ch]
        n = rh * rw
        t = f[bot + rw] - f[top + rw] - f[bot] + f[top]
        t2 = f2[bot + rw] - f2[top + rw] - f2[bot] + f2[top]
        return (n * t2 - t * t) ** 0.5 / n

    def __repr__(self) -> str:
        return f"PreparedPatch({self.h}x{self.w}x{N_CHANNELS}, {N_FEATURE_CHANNELS} feature channels)"


def as_prepared(patch) -> PreparedPatch:
    """PreparedPatch -> itself; raw ndarray -> freshly prepared.

    Idempotent, so it is free on the normal path. The prepare-on-the-fly branch
    exists only so a raw array arriving from TreeFlow or an evaluation helper cannot
    silently produce a WRONG answer — but it costs a full table build, so
    anything hot (training, evaluation, reference routing) must hand over
    already-prepared patches. rgb_forest and rgb_interface see to that.
    """
    return patch if type(patch) is PreparedPatch else PreparedPatch(patch)


def to_uint8_rgb(patch) -> np.ndarray:
    """Displayable (h, w, 3) uint8 view of a patch.

    Accepts a PreparedPatch, a uint8 RGB array, a grayscale array, or a float
    image in either [0, 1] or [0, 255]. This is the RGB replacement for
    polsar_interface.patch_to_rgb — here it is essentially the identity, which
    is the entire point of ImageRF.
    """
    if type(patch) is PreparedPatch:
        return patch.arr
    a = np.asarray(patch)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], N_CHANNELS, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"cannot render an image of shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, N_CHANNELS, axis=2)
    a = a[:, :, :N_CHANNELS]
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if a.size and a.max() <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)
