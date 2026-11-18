"""
phase2_segmentation.py - Phase 2: morphology and kidney segmentation.

Phase 2 covers:
    1. custom structuring elements for erosion, dilation, opening, closing
                                       -> make_disk_structuring_element, make_ellipse_structuring_element,
                                          erode_mask, dilate_mask, open_mask, close_mask
       hole filling, small-object removal -> fill_holes, remove_small_objects, morphology_steps
    2. two classical strategies compared
         Strategy A: multi-Otsu (3+ classes)        -> multi_otsu_thresholds, multi_otsu_class_map,
                                                       segment_multi_otsu_candidates
         Strategy B: seeded region growing from spatial centroids
                                                    -> compute_seed_points, find_kidney_like_seed,
                                                       seeded_region_growing, segment_region_growing_candidates
    3. Dice and IoU                                 -> metrics.py

Pipeline for one slice (segment_slice):
    raw HU  -> find_anatomy (body, bone, spine)          landmarks from RAW HU
    enhanced image -> strategy A or B -> raw mask
    raw mask -> remove bone -> morphology_steps           = "candidates" (all blobs; Phase 3 classifies these)
    candidates -> select_kidney_blobs_by_rules            = final mask (one blob per side)

Everything learned from data (seed offsets, kidney intensity range, position/size
zones, maximum region size) comes from TRAINING ground truth only
(learn_segmentation_priors) and is saved to models/segmentation_priors.json.

Sides: "patient_right" is on the IMAGE LEFT (standard radiology view), and
"patient_left" on the image right.

Textbook references ("Textbook: Ch. N") are to Gonzalez, R. C., & Woods, R. E. (2018).
Digital image processing (4th ed.). Pearson.
"""

import json
from collections import deque

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_multiotsu

import config

SIDES = ("patient_right", "patient_left")

# The 8 neighbours of a pixel as (row step, column step): N, S, W, E and the 4 diagonals.
EIGHT_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# =============================================================================
# 1. Anatomical landmarks (all from RAW HU, not the enhanced image)
# =============================================================================
def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the biggest connected blob of a mask (empty mask stays empty)."""
    labelled, n_blobs = ndimage.label(mask)
    if n_blobs == 0:
        return mask.copy()
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # label 0 is the background
    return labelled == sizes.argmax()


def make_body_mask(hu_slice: np.ndarray, threshold_hu: float = config.BODY_HU_THRESHOLD) -> np.ndarray:
    """Separate the patient's body from the surrounding air and the scanner table.

    Stops thresholds and seeds from landing outside the patient.

    Inputs:
        hu_slice: 2D raw HU slice.
        threshold_hu: pixels above this are tissue (air is about -1000 HU).
    Outputs:
        2D boolean body mask: the largest tissue blob (the table is a separate,
        smaller blob), with internal holes (bowel gas, lungs) filled.
    Textbook: Ch. 10 (Image Segmentation) - global thresholding; Ch. 9 - hole filling.
    """
    tissue = hu_slice > threshold_hu
    return fill_holes(keep_largest_component(tissue))


def make_bone_mask(hu_slice: np.ndarray, body_mask: np.ndarray,
                   threshold_hu: float = config.BONE_HU_THRESHOLD,
                   closing_radius: int = config.BONE_CLOSING_RADIUS,
                   max_hole_area_px: int = config.BONE_MAX_HOLE_AREA_PX) -> np.ndarray:
    """Find bone in RAW HU, including the spongy inside of each vertebra.

    Why raw HU: after windowing, bone and bright contrast-enhanced kidney are both
    white (255), so they cannot be told apart. In raw HU the dense outer shell of
    bone is far above any kidney value (training: 0.01% of kidney pixels > 400 HU).

    Steps:
        1. threshold: HU > threshold_hu  -> the dense bone shell
        2. closing with a small disk      -> bridge small gaps in the shell
        3. fill SMALL enclosed holes      -> the spongy inside of a vertebra (150-300 HU,
           which on its own would look like kidney). Only holes up to max_hole_area_px are
           filled, so a ring of ribs + spine can never fill the whole abdomen.

    Inputs:
        hu_slice: 2D raw HU slice.  body_mask: from make_body_mask.
        threshold_hu, closing_radius, max_hole_area_px: see config (BONE_*).
    Outputs:
        2D boolean bone mask.
    Textbook: Ch. 10 - thresholding; Ch. 9 - closing and hole filling.
    """
    dense_bone = (hu_slice > threshold_hu) & body_mask
    bridged = close_mask(dense_bone, make_disk_structuring_element(closing_radius))

    # Holes = pixels enclosed by bone but not bone themselves. Keep only small ones.
    holes = fill_holes(bridged) & ~bridged
    labelled_holes, _ = ndimage.label(holes)
    hole_sizes = np.bincount(labelled_holes.ravel())
    small_hole_labels = np.flatnonzero((hole_sizes > 0) & (hole_sizes <= max_hole_area_px))
    small_hole_labels = small_hole_labels[small_hole_labels != 0]
    small_holes = np.isin(labelled_holes, small_hole_labels)

    return bridged | small_holes


def find_spine_centroid(bone_mask: np.ndarray, body_mask: np.ndarray,
                        half_width_fraction: float = config.SPINE_SEARCH_HALF_WIDTH_FRACTION) -> tuple[float, float]:
    """Locate the vertebra: the fixed landmark the two kidneys sit either side of.

    Search zone: a vertical band around the body's left-right centre, in the back
    (lower) 60% of the body. The largest bone blob in that zone is the vertebra; its
    centroid is returned. Ribs are outside the band; the aorta (if bright enough to
    count as bone) is much smaller than a vertebra.

    Used to place the region-growing seeds, and as the reference point for the two
    position features in Phase 3.

    Inputs:
        bone_mask: from make_bone_mask.  body_mask: from make_body_mask.
        half_width_fraction: half-width of the search band as a fraction of body width.
    Outputs:
        (row, col) of the spine centroid. If no bone is found, an estimate from the
        body shape (centre column, 75% of the way down) is returned instead.
    """
    body_rows, body_cols = np.nonzero(body_mask)
    top, bottom = body_rows.min(), body_rows.max()
    centre_col = (body_cols.min() + body_cols.max()) / 2
    body_width = body_cols.max() - body_cols.min() + 1

    row_index, col_index = np.indices(bone_mask.shape)
    search_zone = (np.abs(col_index - centre_col) <= half_width_fraction * body_width) & \
                  (row_index >= top + 0.4 * (bottom - top))

    vertebra = keep_largest_component(bone_mask & search_zone)
    if not vertebra.any():
        return (top + 0.75 * (bottom - top), centre_col)  # fallback: typical spine position
    rows, cols = np.nonzero(vertebra)
    return (float(rows.mean()), float(cols.mean()))


def find_anatomy(hu_slice: np.ndarray) -> dict:
    """All landmarks of one slice, computed once and reused by both strategies.

    Outputs:
        dict with "body" (mask), "bone" (mask), "bone_exclusion" (bone grown by a few
        pixels so the bone border is excluded too), "spine" (row, col),
        "body_width" (pixels) and "body_area" (pixels).
    """
    body = make_body_mask(hu_slice)
    bone = make_bone_mask(hu_slice, body)
    body_cols = np.nonzero(body)[1]
    return {
        "body": body,
        "bone": bone,
        "bone_exclusion": dilate_mask(bone, make_disk_structuring_element(config.BONE_EXCLUSION_MARGIN_PX)),
        "spine": find_spine_centroid(bone, body),
        "body_width": int(body_cols.max() - body_cols.min() + 1),
        "body_area": int(body.sum()),
    }


# =============================================================================
# 2. My own structuring elements (explicit numpy arrays, not library presets)
# =============================================================================
def make_disk_structuring_element(radius: int) -> np.ndarray:
    """Build a round (disk) structuring element from the circle equation x^2 + y^2 <= r^2.

    Example, radius 2:
        0 0 1 0 0
        0 1 1 1 0
        1 1 1 1 1
        0 1 1 1 0
        0 0 1 0 0

    A disk treats every direction equally, so it does not bias the kidney outline in
    any direction.

    Inputs:
        radius: disk radius in pixels.
    Outputs:
        (2r+1) x (2r+1) uint8 array of 0/1.
    Textbook: Ch. 9 (Morphological Image Processing) - structuring elements.
    """
    offsets = np.arange(-radius, radius + 1)
    row_offsets, col_offsets = np.meshgrid(offsets, offsets, indexing="ij")
    return (row_offsets ** 2 + col_offsets ** 2 <= radius ** 2).astype(np.uint8)


def make_ellipse_structuring_element(height: int, width: int) -> np.ndarray:
    """Build an elliptical structuring element from (y/a)^2 + (x/b)^2 <= 1.

    Tall-and-narrow, like the kidney's bean shape in axial slices, so smoothing
    follows its natural outline instead of rounding it off.

    Inputs:
        height, width: odd sizes in pixels (config.SE_KIDNEY_ELLIPSE_SIZE).
    Outputs:
        height x width uint8 array of 0/1.
    Textbook: Ch. 9 (Morphological Image Processing) - structuring elements.
    """
    half_height, half_width = height // 2, width // 2
    row_offsets, col_offsets = np.meshgrid(np.arange(-half_height, half_height + 1),
                                           np.arange(-half_width, half_width + 1), indexing="ij")
    # +0.5 so the ends of each axis are included (a pure <= 1 test can drop them on small sizes).
    inside = (row_offsets / (half_height + 0.5)) ** 2 + (col_offsets / (half_width + 0.5)) ** 2 <= 1
    return inside.astype(np.uint8)


def erode_mask(mask: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Erosion: keep a pixel only if the whole SE fits inside the mask there (shrinks objects).

    Inputs:  2D boolean mask, SE (0/1 array).
    Outputs: 2D boolean eroded mask.
    Textbook: Ch. 9 - erosion.
    """
    return ndimage.binary_erosion(mask, structure=structuring_element.astype(bool))


def dilate_mask(mask: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Dilation: add a pixel if the SE, centred there, touches the mask (grows objects).

    Inputs:  2D boolean mask, SE (0/1 array).
    Outputs: 2D boolean dilated mask.
    Textbook: Ch. 9 - dilation.
    """
    return ndimage.binary_dilation(mask, structure=structuring_element.astype(bool))


def open_mask(mask: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Opening = erosion then dilation. Cuts bridges thinner than the SE.

    Used to disconnect the kidney from bowel, liver or spleen where they touch
    through a thin strip of similar intensity.

    Inputs:  2D boolean mask, SE.
    Outputs: 2D boolean opened mask.
    Textbook: Ch. 9 - opening.
    """
    return dilate_mask(erode_mask(mask, structuring_element), structuring_element)


def close_mask(mask: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """Closing = dilation then erosion. Fills gaps and notches narrower than the SE.

    Inputs:  2D boolean mask, SE.
    Outputs: 2D boolean closed mask.
    Textbook: Ch. 9 - closing.
    """
    return erode_mask(dilate_mask(mask, structuring_element), structuring_element)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill fully enclosed holes (e.g. the darker renal sinus inside the kidney outline).

    Inputs:  2D boolean mask.
    Outputs: 2D boolean mask with interior holes filled.
    Textbook: Ch. 9 - hole filling.
    """
    return ndimage.binary_fill_holes(mask)


def remove_small_objects(mask: np.ndarray, min_area_px: int = config.MIN_OBJECT_AREA_PX) -> np.ndarray:
    """Delete connected blobs smaller than min_area_px (noise specks, small vessels).

    Inputs:  2D boolean mask, minimum area in pixels (see config).
    Outputs: 2D boolean mask.
    Textbook: Ch. 9 - connected components.
    """
    labelled, _ = ndimage.label(mask, structure=np.ones((3, 3)))  # 8-connected blobs
    sizes = np.bincount(labelled.ravel())
    big_enough = sizes >= min_area_px
    big_enough[0] = False  # background
    return big_enough[labelled]


def morphology_steps(mask: np.ndarray,
                     opening_radius: int = config.SE_OPENING_RADIUS,
                     closing_radius: int = config.SE_CLOSING_RADIUS,
                     min_area_px: int = config.MIN_OBJECT_AREA_PX) -> dict:
    """The clean-up chain applied to both strategies' raw masks, keeping every step.

    Order, and why:
        1. opening (disk)        cut thin bridges to bowel/liver/spleen FIRST, while
                                 the organs are still separate shapes
        2. remove small objects  drop the specks the opening leaves behind
        3. closing (disk)        close small gaps and notches along the kidney border
        4. fill holes            fill the renal sinus / medulla enclosed by cortex
        5. opening (kidney-shaped ellipse)  smooth the outline along the kidney's shape
        6. remove small objects  again, for pieces the last opening split off

    Inputs:
        mask: 2D boolean raw mask (bone already removed).
        opening_radius, closing_radius, min_area_px: see config.
    Outputs:
        dict of named intermediate masks in order; the last one ("6 final") is the result.
    """
    steps = {"0 raw": mask}
    steps["1 opened"] = open_mask(mask, make_disk_structuring_element(opening_radius))
    steps["2 small removed"] = remove_small_objects(steps["1 opened"], min_area_px)
    steps["3 closed"] = close_mask(steps["2 small removed"], make_disk_structuring_element(closing_radius))
    steps["4 holes filled"] = fill_holes(steps["3 closed"])
    steps["5 ellipse opened"] = open_mask(steps["4 holes filled"],
                                          make_ellipse_structuring_element(*config.SE_KIDNEY_ELLIPSE_SIZE))
    steps["6 final"] = remove_small_objects(steps["5 ellipse opened"], min_area_px)
    return steps


def clean_mask_with_morphology(mask: np.ndarray, opening_radius: int = config.SE_OPENING_RADIUS) -> np.ndarray:
    """Run morphology_steps and return only the final mask."""
    return morphology_steps(mask, opening_radius)["6 final"]


# =============================================================================
# 3. Strategy A - multi-Otsu thresholding
# =============================================================================
def multi_otsu_thresholds(enhanced_image: np.ndarray, region_mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Find n-1 thresholds that split the region's grey levels into n classes.

    Multi-Otsu tries threshold combinations and keeps the one that maximises the
    between-class variance of the histogram, i.e. the classes are as well separated
    as possible.

    Inputs:
        enhanced_image: 2D uint8 slice.
        region_mask: only these pixels build the histogram (inside body, not bone).
        n_classes: number of classes (3 or more).
    Outputs:
        Sorted array of n_classes - 1 thresholds.
    Textbook: Ch. 10 (Image Segmentation) - multilevel thresholding.
    Reference:
        Otsu, N. (1979). A threshold selection method from gray-level histograms.
        IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
        Liao, P.-S., Chen, T.-S., & Chung, P.-C. (2001). A fast algorithm for
        multilevel thresholding. Journal of Information Science and Engineering,
        17(5), 713-727.
    """
    return threshold_multiotsu(enhanced_image[region_mask], classes=n_classes)


def multi_otsu_class_map(enhanced_image: np.ndarray, thresholds: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Label every pixel with its class: 0 = darkest ... n-1 = brightest; -1 outside the region."""
    class_map = np.digitize(enhanced_image, thresholds).astype(np.int8)
    class_map[~region_mask] = -1
    return class_map


def segment_multi_otsu_candidates(enhanced_image: np.ndarray, anatomy: dict,
                                  n_classes: int = config.MULTI_OTSU_CLASSES,
                                  n_kidney_classes: int = config.MULTI_OTSU_KIDNEY_CLASSES) -> np.ndarray:
    """Strategy A raw mask: the brightest n_kidney_classes classes inside the body, minus bone.

    Contrast-enhanced kidney is among the brightest soft tissue, so the kidney lies
    in the top class(es) - together with other bright structures (vessels, bowel
    with oral contrast, sometimes liver/spleen). All of them come out here as
    candidate blobs; the rule filter (or Phase 3's k-NN) picks the kidneys.

    Inputs:
        enhanced_image: 2D uint8 output of phase1.enhance_slice.
        anatomy: from find_anatomy.
        n_classes, n_kidney_classes: see config (chosen on training Dice).
    Outputs:
        2D boolean raw mask (before morphology).
    """
    # Histogram only from soft tissue: air outside the body and bone would steal thresholds.
    region = anatomy["body"] & ~anatomy["bone_exclusion"]
    thresholds = multi_otsu_thresholds(enhanced_image, region, n_classes)
    class_map = multi_otsu_class_map(enhanced_image, thresholds, region)
    return class_map >= n_classes - n_kidney_classes


# =============================================================================
# 4. Learning priors from TRAINING ground truth
# =============================================================================
def blob_position(blob_mask: np.ndarray, anatomy: dict) -> dict:
    """Where a blob sits relative to the spine, scaled by body width (so patient size cancels out).

        dx = (blob column - spine column) / body width   (negative = image left = patient's right)
        dy = (blob row    - spine row)    / body width   (negative = in front of the spine)
        area = blob pixels / body pixels
    """
    rows, cols = np.nonzero(blob_mask)
    spine_row, spine_col = anatomy["spine"]
    return {
        "dx": (cols.mean() - spine_col) / anatomy["body_width"],
        "dy": (rows.mean() - spine_row) / anatomy["body_width"],
        "area": rows.size / anatomy["body_area"],
        "side": SIDES[0] if cols.mean() < spine_col else SIDES[1],
    }


def ground_truth_kidney_per_side(ground_truth: np.ndarray, anatomy: dict) -> dict:
    """The largest ground-truth kidney piece on each side of the spine (None if that side has none)."""
    spine_col = anatomy["spine"][1]
    column_index = np.indices(ground_truth.shape)[1]
    result = {}
    for side, side_zone in zip(SIDES, [column_index < spine_col, column_index >= spine_col]):
        piece = keep_largest_component(ground_truth & side_zone)
        result[side] = piece if piece.sum() >= config.MIN_OBJECT_AREA_PX else None
    return result


def kidney_intensity_range(training_slices: list[dict], images: list[np.ndarray],
                           percentiles: tuple[float, float] = config.KIDNEY_INTENSITY_PERCENTILES) -> list[float]:
    """Grey-level range of ground-truth kidney pixels over the training slices.

    A seed is accepted as kidney-like only inside this range. The low percentile is
    set fairly high on purpose (chosen by training Dice): it pushes seeds onto the
    bright outer cortex rather than the darker inner kidney. The cortex usually forms
    a closed ring, so growing the ring and then filling holes captures the whole kidney.

    Inputs:
        training_slices: training slice dicts (with "ground_truth").
        images: the matching enhanced images.
        percentiles: (low, high) percentiles of kidney grey levels.
    Outputs:
        [low grey level, high grey level].
    """
    kidney_values = np.concatenate([image[s["ground_truth"]] for image, s in zip(images, training_slices)])
    low, high = np.percentile(kidney_values, percentiles)
    return [float(low), float(high)]


def learn_segmentation_priors(training_slices: list[dict], enhanced_by_input: dict) -> dict:
    """Learn, from TRAINING ground truth only, everything the strategies need to know.

    Per side of the spine:
        seed offset   - median kidney position (dx, dy) -> where seeds are placed
        position zone - min..max of dx and dy, widened by config.ZONE_MARGIN_FRACTION
        size zone     - min..max kidney area (fraction of body), widened the same way
        spread        - std of dx and dy, used to rank blobs by closeness to the usual spot
    Per enhancement input (none / CLAHE / HE):
        kidney intensity range - percentiles of ground-truth kidney grey levels (seed check)
    Overall:
        maximum region size for region growing = largest training kidney x config factor

    Inputs:
        training_slices: dicts from data_loading.load_split("train"), each with "anatomy".
        enhanced_by_input: {input name: list of enhanced images, same order as the slices}.
    Outputs:
        The priors dict (also what save_priors writes).
    """
    positions = {side: [] for side in SIDES}
    for slice_data in training_slices:
        for side, piece in ground_truth_kidney_per_side(slice_data["ground_truth"], slice_data["anatomy"]).items():
            if piece is not None:
                positions[side].append(blob_position(piece, slice_data["anatomy"]))

    priors = {"sides": {}, "intensity_range": {}}
    for side in SIDES:
        values = {key: np.array([p[key] for p in positions[side]]) for key in ("dx", "dy", "area")}
        side_priors = {"n_training_kidneys": len(positions[side]),
                       "seed_dx": float(np.median(values["dx"])), "seed_dy": float(np.median(values["dy"])),
                       "std_dx": float(values["dx"].std()), "std_dy": float(values["dy"].std())}
        for key in ("dx", "dy", "area"):
            low, high = values[key].min(), values[key].max()
            margin = config.ZONE_MARGIN_FRACTION * (high - low)
            side_priors[f"{key}_min"] = float(max(low - margin, 0) if key == "area" else low - margin)
            side_priors[f"{key}_max"] = float(high + margin)
        priors["sides"][side] = side_priors

    for input_name, images in enhanced_by_input.items():
        priors["intensity_range"][input_name] = kidney_intensity_range(training_slices, images)

    largest_kidney = max(priors["sides"][side]["area_max"] for side in SIDES)
    priors["max_region_area_fraction"] = float(largest_kidney * config.REGION_GROW_MAX_AREA_FACTOR)
    return priors


def save_priors(priors: dict) -> None:
    """Write the learned priors to config.SEGMENTATION_PRIORS_FILE (models/segmentation_priors.json)."""
    config.SEGMENTATION_PRIORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SEGMENTATION_PRIORS_FILE.write_text(json.dumps(priors, indent=2), encoding="utf-8")


def load_priors() -> dict:
    """Read the learned priors written by scripts/step3_compare_segmentation.py."""
    return json.loads(config.SEGMENTATION_PRIORS_FILE.read_text(encoding="utf-8"))


# =============================================================================
# 5. Strategy B - seeded region growing
# =============================================================================
def find_kidney_like_seed(enhanced_image: np.ndarray, expected: tuple[int, int], intensity_range: list[float],
                          blocked: np.ndarray, search_radius: int) -> tuple[tuple[int, int] | None, str]:
    """Check a seed's grey level; if it is not kidney-like, look nearby for one that is.

    Inputs:
        enhanced_image: 2D uint8 slice.
        expected: (row, col) where the learned offset puts the seed.
        intensity_range: [low, high] kidney grey levels learned from training.
        blocked: pixels that may never be seeds (outside body, bone).
        search_radius: how far (pixels) to look for a better seed.
    Outputs:
        (seed or None, status) - status is "ok" (expected point was fine), "moved"
        (nearest kidney-like pixel within the radius) or "skipped" (none found:
        often this side simply has no kidney on this slice).
    """
    low, high = intensity_range
    row, col = expected
    rows, cols = enhanced_image.shape
    if 0 <= row < rows and 0 <= col < cols and not blocked[row, col] and low <= enhanced_image[row, col] <= high:
        return (row, col), "ok"

    # Search a disk around the expected point for kidney-like, unblocked pixels.
    row0, row1 = max(row - search_radius, 0), min(row + search_radius + 1, rows)
    col0, col1 = max(col - search_radius, 0), min(col + search_radius + 1, cols)
    window = enhanced_image[row0:row1, col0:col1]
    window_rows, window_cols = np.indices(window.shape)
    distance = np.hypot(window_rows + row0 - row, window_cols + col0 - col)
    acceptable = (window >= low) & (window <= high) & ~blocked[row0:row1, col0:col1] & (distance <= search_radius)
    if not acceptable.any():
        return None, "skipped"
    nearest = np.argmin(np.where(acceptable, distance, np.inf))
    nearest_row, nearest_col = np.unravel_index(nearest, window.shape)
    return (int(nearest_row + row0), int(nearest_col + col0)), "moved"


def expected_seed_position(anatomy: dict, priors: dict, side: str) -> tuple[int, int]:
    """Where the learned offset places this side's seed: spine centroid + offset x body width."""
    spine_row, spine_col = anatomy["spine"]
    side_priors = priors["sides"][side]
    return (int(round(spine_row + side_priors["seed_dy"] * anatomy["body_width"])),
            int(round(spine_col + side_priors["seed_dx"] * anatomy["body_width"])))


def compute_seed_points(enhanced_image: np.ndarray, anatomy: dict, priors: dict, input_name: str) -> list[dict]:
    """Place one seed per side: spine centroid + learned offset, then check it.

    Inputs:
        enhanced_image: 2D uint8 slice.  anatomy: from find_anatomy.
        priors: from learn_segmentation_priors.  input_name: which intensity range to use.
    Outputs:
        List of dicts, one per side: {"side", "expected", "seed" (or None), "status"}.
    """
    blocked = ~anatomy["body"] | anatomy["bone_exclusion"]
    search_radius = int(round(config.SEED_SEARCH_RADIUS_FRACTION * anatomy["body_width"]))
    seeds = []
    for side in SIDES:
        expected = expected_seed_position(anatomy, priors, side)
        seed, status = find_kidney_like_seed(enhanced_image, expected, priors["intensity_range"][input_name],
                                             blocked, search_radius)
        seeds.append({"side": side, "expected": expected, "seed": seed, "status": status})
    return seeds


def seeded_region_growing(image: np.ndarray, seed: tuple[int, int], tolerance: float,
                          max_area_px: int, blocked: np.ndarray) -> np.ndarray:
    """Grow a region outward from one seed, adding 8-connected neighbours of similar grey level.

    A queue holds pixels whose neighbours still need checking. A neighbour joins
    the region if |neighbour - current region mean| < tolerance; the region mean is
    updated every time a pixel joins. Each pixel is tested once. Growth stops when
    the queue is empty or the region reaches max_area_px (a region that big has
    leaked into another organ - the cap stops it swallowing the whole abdomen).

    This is the simple first-in-first-out variant; Adams & Bischof's original
    processes pixels in order of similarity using a sorted queue.

    Inputs:
        image: 2D uint8 enhanced slice.
        seed: (row, col) start pixel.
        tolerance: maximum allowed difference from the region mean (grey levels).
        max_area_px: size cap in pixels.
        blocked: pixels that may never join (outside body, bone).
    Outputs:
        2D boolean mask of the grown region.
    Reference:
        Adams, R., & Bischof, L. (1994). Seeded region growing. IEEE Transactions on
        Pattern Analysis and Machine Intelligence, 16(6), 641-647.
    """
    rows, cols = image.shape
    # Python lists are much faster than numpy for reading one pixel at a time.
    pixels = image.tolist()
    already_tested = blocked.tolist()  # blocked pixels count as "tested and rejected"

    seed_row, seed_col = seed
    region = [(seed_row, seed_col)]
    region_sum = float(pixels[seed_row][seed_col])
    already_tested[seed_row][seed_col] = True
    queue = deque([(seed_row, seed_col)])

    while queue and len(region) < max_area_px:
        row, col = queue.popleft()
        for row_step, col_step in EIGHT_NEIGHBOURS:
            next_row, next_col = row + row_step, col + col_step
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if already_tested[next_row][next_col]:
                continue
            already_tested[next_row][next_col] = True

            region_mean = region_sum / len(region)
            if abs(pixels[next_row][next_col] - region_mean) < tolerance:
                region.append((next_row, next_col))
                region_sum += pixels[next_row][next_col]
                queue.append((next_row, next_col))

    mask = np.zeros(image.shape, dtype=bool)
    region_rows, region_cols = zip(*region)
    mask[list(region_rows), list(region_cols)] = True
    return mask


def segment_region_growing_candidates(enhanced_image: np.ndarray, anatomy: dict, priors: dict, input_name: str,
                                      tolerance: float = config.REGION_GROW_TOLERANCE) -> tuple[np.ndarray, list[dict]]:
    """Strategy B raw mask: grow from each accepted seed and combine the regions.

    Inputs:
        enhanced_image: 2D uint8 slice.  anatomy: from find_anatomy.
        priors: learned priors.  input_name: which intensity range to use for the seed check.
        tolerance: see config.REGION_GROW_TOLERANCE.
    Outputs:
        (raw mask before morphology, list of seed dicts from compute_seed_points).
    """
    seeds = compute_seed_points(enhanced_image, anatomy, priors, input_name)
    blocked = ~anatomy["body"] | anatomy["bone_exclusion"]
    max_area_px = int(priors["max_region_area_fraction"] * anatomy["body_area"])

    combined = np.zeros(enhanced_image.shape, dtype=bool)
    for seed_info in seeds:
        if seed_info["seed"] is not None:
            combined |= seeded_region_growing(enhanced_image, seed_info["seed"], tolerance, max_area_px, blocked)
    return combined, seeds


# =============================================================================
# 6. Rule filter: one kidney blob per side (Phase 3's k-NN will replace this)
# =============================================================================
def distance_from_usual_kidney_position(blob_mask: np.ndarray, anatomy: dict, priors: dict) -> float:
    """How far a blob's centroid is from the usual kidney position on its side, in training standard deviations."""
    position = blob_position(blob_mask, anatomy)
    zone = priors["sides"][position["side"]]
    return float(np.hypot((position["dx"] - zone["seed_dx"]) / zone["std_dx"],
                          (position["dy"] - zone["seed_dy"]) / zone["std_dy"]))


def select_kidney_blobs_by_rules(candidate_mask: np.ndarray, anatomy: dict, priors: dict) -> np.ndarray:
    """Keep at most one blob per side of the spine: the one that looks most like a kidney position.

    A blob is allowed on a side only if its centroid offset (dx, dy) and its area lie
    inside that side's zone learned from training. Among allowed blobs, the one
    closest to the usual kidney position (distance measured in training standard
    deviations) is kept.

    A simple, explainable alternative to the Phase 3 classifier.

    Inputs:
        candidate_mask: 2D boolean mask after morphology (may hold many blobs).
        anatomy: from find_anatomy.  priors: learned priors.
    Outputs:
        2D boolean mask with at most two blobs.
    """
    labelled, n_blobs = ndimage.label(candidate_mask, structure=np.ones((3, 3)))
    best_per_side = {side: (np.inf, None) for side in SIDES}

    for blob_label in range(1, n_blobs + 1):
        blob = labelled == blob_label
        position = blob_position(blob, anatomy)
        zone = priors["sides"][position["side"]]
        inside_zone = (zone["dx_min"] <= position["dx"] <= zone["dx_max"]
                       and zone["dy_min"] <= position["dy"] <= zone["dy_max"]
                       and zone["area_min"] <= position["area"] <= zone["area_max"])
        if not inside_zone:
            continue
        distance = distance_from_usual_kidney_position(blob, anatomy, priors)
        if distance < best_per_side[position["side"]][0]:
            best_per_side[position["side"]] = (distance, blob_label)

    keep_labels = [label for _, label in best_per_side.values() if label is not None]
    return np.isin(labelled, keep_labels)


# =============================================================================
# 7. One slice end to end
# =============================================================================
def clean_candidates(raw_mask: np.ndarray, anatomy: dict, opening_radius: int = config.SE_OPENING_RADIUS) -> np.ndarray:
    """Remove bone and run the morphology chain: the candidate blobs Phase 3's k-NN classifies."""
    return morphology_steps(raw_mask & ~anatomy["bone_exclusion"], opening_radius)["6 final"]


def finish_segmentation(raw_mask: np.ndarray, anatomy: dict, priors: dict,
                        opening_radius: int = config.SE_OPENING_RADIUS) -> dict:
    """Common ending for both strategies: remove bone, clean with morphology, pick kidney blobs.

    Outputs:
        dict with "raw" (bone removed), "steps" (morphology stages), "candidates"
        (all blobs after morphology - what Phase 3 classifies) and "final" (rule-filtered).
    """
    # Bone is never kidney: after windowing both are white, so remove it explicitly.
    raw = raw_mask & ~anatomy["bone_exclusion"]
    steps = morphology_steps(raw, opening_radius)
    candidates = steps["6 final"]
    return {"raw": raw, "steps": steps, "candidates": candidates,
            "final": select_kidney_blobs_by_rules(candidates, anatomy, priors)}


def segment_slice(enhanced_image: np.ndarray, anatomy: dict, priors: dict, strategy: str, input_name: str,
                  **settings) -> dict:
    """Run one strategy on one slice and return every stage.

    Inputs:
        enhanced_image: 2D uint8 slice from phase1.
        anatomy: from find_anatomy.  priors: learned priors.
        strategy: "multi_otsu" or "region_growing".
        input_name: "none", "clahe" or "histeq" (selects the learned intensity range).
        settings: optional overrides - n_classes, n_kidney_classes, tolerance, opening_radius.
    Outputs:
        dict with "raw" (strategy output minus bone), "steps" (morphology stages),
        "candidates" (all blobs after morphology - what Phase 3 classifies),
        "final" (rule-filtered mask) and "seeds" (region growing only, else []).
    """
    seeds = []
    if strategy == "multi_otsu":
        raw = segment_multi_otsu_candidates(enhanced_image, anatomy,
                                            settings.get("n_classes", config.MULTI_OTSU_CLASSES),
                                            settings.get("n_kidney_classes", config.MULTI_OTSU_KIDNEY_CLASSES))
    elif strategy == "region_growing":
        raw, seeds = segment_region_growing_candidates(enhanced_image, anatomy, priors, input_name,
                                                       settings.get("tolerance", config.REGION_GROW_TOLERANCE))
    else:
        raise ValueError('strategy must be "multi_otsu" or "region_growing"')

    result = finish_segmentation(raw, anatomy, priors, settings.get("opening_radius", config.SE_OPENING_RADIUS))
    result["seeds"] = seeds
    return result
