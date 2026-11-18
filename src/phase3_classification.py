"""
phase3_classification.py - Phase 3: feature extraction and k-NN classification.

Task: every candidate region produced by Phase 2 (each multi-Otsu blob and each
region-growing region) is described by 17 features and classified as kidney (1)
or not kidney (0). Training labels come from overlap with the ground truth.

Phase 3 covers:
    1. features   -> shape_features (6), glcm_features_inside_blob (5),
                     first_order_hu_features (4), position_features_relative_to_spine (2)
    2. normalised feature vectors, classical classifier, 70/30 split, 5-fold CV
                  -> fit_feature_scaler, tune_k_by_grouped_cross_validation, train_knn_classifier
    3. confusion matrix, accuracy, recall, specificity, F1 -> metrics.py

The classifier is then used inside the segmentation (segment_with_knn):
    Strategy A: keep the multi-Otsu blobs the k-NN calls kidney (best one per side).
    Strategy B: the k-NN checks each grown region; if rejected, the next-best seed
                on that side is tried (up to config.SEED_MAX_ATTEMPTS).

Textbook references ("Textbook: Ch. N") are to Gonzalez, R. C., & Woods, R. E. (2018).
Digital image processing (4th ed., US edition). Pearson.
"""

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import phase1_enhancement, phase2_segmentation

# The 17 features in a fixed order (this order is the column order of the feature matrix).
SHAPE_FEATURE_NAMES = ["area", "perimeter_to_area", "circularity", "compactness", "eccentricity", "solidity"]
GLCM_FEATURE_NAMES = ["glcm_energy", "glcm_contrast", "glcm_homogeneity", "glcm_correlation", "glcm_entropy"]
FIRST_ORDER_FEATURE_NAMES = ["hu_mean", "hu_variance", "hu_skewness", "hu_kurtosis"]
POSITION_FEATURE_NAMES = ["dx_from_spine", "dy_from_spine"]
FEATURE_NAMES = SHAPE_FEATURE_NAMES + GLCM_FEATURE_NAMES + FIRST_ORDER_FEATURE_NAMES + POSITION_FEATURE_NAMES


# =============================================================================
# 1. Candidates and their labels
# =============================================================================
def extract_candidate_blobs(segmentation_mask: np.ndarray,
                            min_area_px: int = config.MIN_CANDIDATE_AREA_PX) -> list[np.ndarray]:
    """Split a mask into separate 8-connected blobs (the candidates), ignoring tiny ones.

    Inputs:
        segmentation_mask: 2D boolean mask after Phase 2 morphology.
        min_area_px: ignore smaller blobs (see config).
    Outputs:
        List of 2D boolean masks, one per blob.
    Textbook: Ch. 9 (Morphological Image Processing) - connected components.
    """
    labelled, n_blobs = ndimage.label(segmentation_mask, structure=np.ones((3, 3)))
    blobs = []
    for blob_label in range(1, n_blobs + 1):
        blob = labelled == blob_label
        if blob.sum() >= min_area_px:
            blobs.append(blob)
    return blobs


def label_blob_against_ground_truth(blob_mask: np.ndarray, ground_truth_mask: np.ndarray,
                                    overlap_fraction: float = config.BLOB_KIDNEY_OVERLAP_FRACTION) -> int:
    """True class of a candidate: 1 if at least half of it lies inside the real kidney, else 0.

    These are the labels the classifier is trained and tested against.

    Inputs:
        blob_mask, ground_truth_mask: 2D boolean masks.
        overlap_fraction: see config.
    Outputs:
        1 (kidney) or 0 (not kidney).
    """
    inside = np.logical_and(blob_mask, ground_truth_mask).sum() / blob_mask.sum()
    return int(inside >= overlap_fraction)


# =============================================================================
# 2. The 17 features
# =============================================================================
def shape_features(blob_mask: np.ndarray, anatomy: dict) -> dict:
    """Six shape descriptors of one blob.

        area              - blob pixels / body pixels (a fraction, so pixel size and
                            patient size cancel out; segment_kidney gets no pixel spacing)
        perimeter_to_area - perimeter / area in pixels (high for thin or ragged shapes)
        circularity       - 4*pi*area / perimeter^2 (1 = perfect circle)
        compactness       - perimeter^2 / area. Note: this equals 4*pi / circularity,
                            so the two carry the same information in different units.
        eccentricity      - 0 = circle, close to 1 = elongated (from the fitted ellipse)
        solidity          - area / convex hull area (1 = no dents; the kidney's hilum lowers it)

    Inputs:  2D boolean blob mask, anatomy (for the body area).
    Outputs: dict with the 6 names in SHAPE_FEATURE_NAMES.
    Textbook: Ch. 12 (Feature Extraction) - regional descriptors.
    """
    region = regionprops(blob_mask.astype(np.uint8))[0]
    area_px = float(region.area)
    perimeter = max(float(region.perimeter), 1.0)  # a 1-pixel blob would otherwise divide by zero
    return {
        "area": area_px / anatomy["body_area"],
        "perimeter_to_area": perimeter / area_px,
        "circularity": 4 * np.pi * area_px / perimeter ** 2,
        "compactness": perimeter ** 2 / area_px,
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
    }


def glcm_features_inside_blob(windowed_image: np.ndarray, blob_mask: np.ndarray,
                              levels: int = config.GLCM_LEVELS,
                              distances: list[int] = config.GLCM_DISTANCES,
                              angles_degrees: list[int] = config.GLCM_ANGLES_DEGREES) -> dict:
    """Five GLCM texture features, counting ONLY pixel pairs where both pixels are inside the blob.

    Computed on the WINDOWED image (before diffusion), because anisotropic diffusion
    flattens the kidney's texture into smooth patches.

    How "inside the blob only" works: quantise blob pixels to grey levels 1..levels,
    set every outside pixel to 0, build the co-occurrence matrix with levels+1 grey
    levels, then delete row 0 and column 0 - so no pair involving an outside pixel
    is counted. The matrix is symmetric (each pair counted both ways) and normalised
    to probabilities p(i, j). Each feature is averaged over 0, 45, 90 and 135 degrees,
    so it does not depend on the blob's orientation.

        energy      = sum p^2                         (high = few, repeated grey-level pairs)
        contrast    = sum p * (i - j)^2               (high = big jumps between neighbours)
        homogeneity = sum p / (1 + (i - j)^2)         (high = neighbours similar)
        correlation = how linearly neighbour grey levels depend on each other
        entropy     = -sum p * log2(p)  (by hand)     (high = disordered texture)

    Inputs:
        windowed_image: 2D uint8 windowed slice.  blob_mask: 2D boolean blob.
        levels, distances, angles_degrees: see config (GLCM_*).
    Outputs:
        dict with the 5 names in GLCM_FEATURE_NAMES.
    Textbook: Ch. 12 (Feature Extraction) - co-occurrence matrix.
    Reference:
        Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for
        image classification. IEEE Transactions on Systems, Man, and Cybernetics,
        SMC-3(6), 610-621.
    """
    # Crop to the blob's bounding box (plus 1 pixel) so the matrix is quick to build.
    rows, cols = np.nonzero(blob_mask)
    row0, row1 = max(rows.min() - 1, 0), rows.max() + 2
    col0, col1 = max(cols.min() - 1, 0), cols.max() + 2
    image_crop = windowed_image[row0:row1, col0:col1].astype(np.int32)
    blob_crop = blob_mask[row0:row1, col0:col1]

    # Grey levels 0..255 -> 1..levels inside the blob; 0 marks "outside".
    quantised = (image_crop * levels // 256 + 1) * blob_crop
    co_occurrence = graycomatrix(quantised.astype(np.uint8), distances=distances,
                                 angles=np.deg2rad(angles_degrees), levels=levels + 1, symmetric=True)
    co_occurrence = co_occurrence[1:, 1:, :, :].astype(np.float64)  # drop pairs touching "outside"

    # Normalise each (distance, angle) matrix so it sums to 1 (skip empty ones: tiny blobs).
    totals = co_occurrence.sum(axis=(0, 1), keepdims=True)
    probabilities = np.divide(co_occurrence, totals, out=np.zeros_like(co_occurrence), where=totals > 0)

    features = {}
    for name in ("energy", "contrast", "homogeneity", "correlation"):
        # A perfectly uniform blob has zero variance, so correlation is undefined (NaN): use 0.
        features[f"glcm_{name}"] = float(np.nan_to_num(graycoprops(probabilities, name)).mean())

    # Entropy by hand, per angle, then averaged. 0 * log(0) is treated as 0.
    nonzero = probabilities > 0
    entropy_terms = np.zeros_like(probabilities)
    entropy_terms[nonzero] = probabilities[nonzero] * np.log2(probabilities[nonzero])
    features["glcm_entropy"] = float(-entropy_terms.sum(axis=(0, 1)).mean())
    return features


def first_order_hu_features(hu_slice: np.ndarray, blob_mask: np.ndarray) -> dict:
    """Four first-order statistics of the RAW HU values inside the blob.

    Raw HU (not an enhanced image) because HU have physical meaning: enhancing
    kidney, liver, bowel and vessels have typical density ranges that windowing and
    equalisation would distort.

        mean, variance                 - typical density and its spread
        skewness                       - asymmetry (a long tail of bright or dark pixels)
        kurtosis (excess, normal = 0)  - how heavy the tails are

    Inputs:  2D raw HU slice, 2D boolean blob mask.
    Outputs: dict with the 4 names in FIRST_ORDER_FEATURE_NAMES.
    Textbook: Ch. 12 (Feature Extraction) - statistical moments.
    """
    values = hu_slice[blob_mask].astype(np.float64)
    return {
        "hu_mean": float(values.mean()),
        "hu_variance": float(values.var()),
        # A constant region has no defined skew/kurtosis (NaN): use 0, the normal-distribution value.
        "hu_skewness": float(np.nan_to_num(stats.skew(values))),
        "hu_kurtosis": float(np.nan_to_num(stats.kurtosis(values))),
    }


def position_features_relative_to_spine(blob_mask: np.ndarray, anatomy: dict) -> dict:
    """Where the blob sits relative to the spine centroid, scaled by body width.

        dx_from_spine - |column offset| / body width. Absolute value, so the left and
                        right kidneys (which sit a similar distance either side) look alike.
        dy_from_spine - row offset / body width (negative = in front of the spine).

    Scaling by body width cancels patient size and pixel size.

    Position separates the kidneys from liver, spleen and bowel, which can have
    similar density and texture.

    Inputs:  2D boolean blob, anatomy (spine centroid and body width).
    Outputs: dict with dx_from_spine, dy_from_spine.
    """
    position = phase2_segmentation.blob_position(blob_mask, anatomy)
    return {"dx_from_spine": abs(position["dx"]), "dy_from_spine": position["dy"]}


def extract_blob_features(hu_slice: np.ndarray, windowed_image: np.ndarray, blob_mask: np.ndarray,
                          anatomy: dict) -> dict:
    """All 17 features for one blob, in FEATURE_NAMES order.

    Inputs: raw HU slice, windowed uint8 slice, blob mask, anatomy.
    Outputs: dict of 17 feature values.
    """
    features = {}
    features.update(shape_features(blob_mask, anatomy))
    features.update(glcm_features_inside_blob(windowed_image, blob_mask))
    features.update(first_order_hu_features(hu_slice, blob_mask))
    features.update(position_features_relative_to_spine(blob_mask, anatomy))
    return {name: features[name] for name in FEATURE_NAMES}


def features_for_blobs(blobs: list[np.ndarray], hu_slice: np.ndarray, windowed_image: np.ndarray,
                       anatomy: dict) -> pd.DataFrame:
    """Feature table (one row per blob, the 17 FEATURE_NAMES as columns); empty table if no blobs.

    All 17 features are always computed (they are all reported in the feature
    analysis); the classifier then uses only config.KNN_FEATURES.
    """
    rows = [extract_blob_features(hu_slice, windowed_image, blob, anatomy) for blob in blobs]
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=np.float64)


# =============================================================================
# 3. Classifier: scaler + k-NN
# =============================================================================
def fit_feature_scaler(training_features: pd.DataFrame) -> StandardScaler:
    """Learn each feature's mean and standard deviation on the TRAINING candidates only.

    Why k-NN needs this: k-NN decides by Euclidean DISTANCE between feature vectors.
    Without standardising, a feature with large numbers (HU variance in the
    thousands, compactness around 20-100) would dominate the distance and features
    with small numbers (eccentricity 0-1, area fraction 0.001-0.05) would be ignored.
    z = (x - mean) / std puts every feature on the same scale, so each one counts.

    Inputs:  training feature table, ONLY the columns the classifier uses (config.KNN_FEATURES).
             Fitting on a table makes the scaler remember the column names, so a saved
             model can never be given the wrong features later.
    Outputs: fitted sklearn StandardScaler.
    """
    return StandardScaler().fit(training_features)


def train_knn_classifier(standardised_features: np.ndarray, labels: np.ndarray,
                         k: int = config.KNN_K, weights: str = config.KNN_WEIGHTS) -> KNeighborsClassifier:
    """Fit the k-NN on standardised training features.

    k-NN stores the training vectors. A new candidate gets the majority class of its
    k closest training candidates (Euclidean distance). With weights="distance",
    closer neighbours get a bigger vote (weight 1 / distance).

    Inputs:  standardised training features, labels (0/1), k, weighting.
    Outputs: fitted sklearn KNeighborsClassifier.
    Textbook: Ch. 13 (Image Pattern Classification) - minimum-distance classifier, which k-NN generalises.
    Reference: Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
               IEEE Transactions on Information Theory, 13(1), 21-27.
    """
    return KNeighborsClassifier(n_neighbors=k, weights=weights, metric=config.KNN_METRIC).fit(
        standardised_features, labels)


def make_knn_pipeline(k: int, weights: str) -> Pipeline:
    """Scaler + k-NN as one object, so cross-validation refits the scaler inside every fold."""
    return Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, metric=config.KNN_METRIC))])


def grouped_cv_scores(features: pd.DataFrame, labels: np.ndarray, patient_ids: np.ndarray,
                      k: int = config.KNN_K, weights: str = config.KNN_WEIGHTS) -> dict:
    """Mean and std of F1 and accuracy over 5 folds that each hold out whole patients."""
    scores = cross_validate(make_knn_pipeline(k, weights), features, labels, groups=patient_ids,
                            cv=GroupKFold(n_splits=config.CV_FOLDS), scoring=["f1", "accuracy"])
    return {"f1_mean": scores["test_f1"].mean(), "f1_std": scores["test_f1"].std(),
            "accuracy_mean": scores["test_accuracy"].mean(), "accuracy_std": scores["test_accuracy"].std()}


def backward_feature_selection(features: pd.DataFrame, labels: np.ndarray, patient_ids: np.ndarray,
                               starting_features: list[str], k: int = config.KNN_K,
                               weights: str = config.KNN_WEIGHTS) -> tuple[list[str], pd.DataFrame]:
    """Remove features one at a time while the grouped cross-validated F1 keeps improving.

    Each round tries removing every remaining feature (one at a time), measures CV F1
    without it, and removes the one whose removal gives the highest F1 - but only if
    that F1 beats the current set. Stops when no single removal improves F1.

    Why: k-NN weighs every feature equally in its distance, so a feature that does
    not help separate kidney from not-kidney only adds noise to the distance.
    Caution: with 8 training patients, CV F1 itself is noisy (fold std about 0.1), so
    small improvements may be luck; the untouched test set is the real check.

    Inputs:
        features: training feature table (all columns).  labels, patient_ids: per candidate.
        starting_features: the features to start from.  k, weights: k-NN settings used.
    Outputs:
        (selected feature names, log table: one row per tried removal per round).
    """
    selected = list(starting_features)
    current = grouped_cv_scores(features[selected], labels, patient_ids, k, weights)
    log = [{"round": 0, "tried_removing": "(none - starting set)", "n_features": len(selected),
            **current, "removed": False}]
    round_number = 0
    while len(selected) > 1:
        round_number += 1
        trials = []
        for feature in selected:
            remaining = [name for name in selected if name != feature]
            trials.append((feature, grouped_cv_scores(features[remaining], labels, patient_ids, k, weights)))
        best_feature, best_scores = max(trials, key=lambda trial: trial[1]["f1_mean"])
        improved = best_scores["f1_mean"] > current["f1_mean"]
        for feature, scores in trials:
            log.append({"round": round_number, "tried_removing": feature, "n_features": len(selected) - 1,
                        **scores, "removed": improved and feature == best_feature})
        if not improved:
            break
        selected.remove(best_feature)
        current = best_scores
    return selected, pd.DataFrame(log)


def tune_k_by_grouped_cross_validation(features: pd.DataFrame, labels: np.ndarray, patient_ids: np.ndarray,
                                       k_candidates: list[int] = config.KNN_K_CANDIDATES,
                                       weight_candidates: list[str] = config.KNN_WEIGHT_CANDIDATES,
                                       n_folds: int = config.CV_FOLDS) -> tuple[dict, pd.DataFrame]:
    """Choose k and the weighting with 5-fold cross-validation where each fold holds out whole patients.

    Grouping by patient matters: candidates from one patient (neighbouring slices,
    the same kidney found by both strategies) are near-duplicates. If they could sit
    in both the training and validation part of a fold, k-NN would find its own
    "twin" and the score would be far too optimistic.

    The scaler is inside the pipeline, so it is refitted on each fold's training part
    and the held-out patients never influence the standardisation.

    Inputs:
        features: training feature table (the classifier's columns), NOT yet standardised.
        labels: 0/1 per candidate.  patient_ids: case_id per candidate (the groups).
        k_candidates, weight_candidates, n_folds: see config.
    Outputs:
        (best settings {"k", "weights"}, table with mean and std F1 and accuracy for every setting).
        The winner has the highest mean F1 (config.CV_SCORING), because the classes are imbalanced.
    Reference:
        Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy
        estimation and model selection. Proceedings of the 14th International Joint
        Conference on Artificial Intelligence (IJCAI), 1137-1145.
    """
    search = GridSearchCV(make_knn_pipeline(1, "uniform"),
                          param_grid={"knn__n_neighbors": k_candidates, "knn__weights": weight_candidates},
                          scoring={"f1": "f1", "accuracy": "accuracy"}, refit=config.CV_SCORING,
                          cv=GroupKFold(n_splits=n_folds))
    search.fit(features, labels, groups=patient_ids)

    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "k": results["param_knn__n_neighbors"].astype(int),
        "weights": results["param_knn__weights"],
        "f1_mean": results["mean_test_f1"], "f1_std": results["std_test_f1"],
        "accuracy_mean": results["mean_test_accuracy"], "accuracy_std": results["std_test_accuracy"],
    }).sort_values(["weights", "k"]).reset_index(drop=True)
    best = {"k": int(search.best_params_["knn__n_neighbors"]), "weights": search.best_params_["knn__weights"]}
    return best, table


def save_classifier(scaler: StandardScaler, classifier: KNeighborsClassifier) -> None:
    """Save the fitted scaler and k-NN to models/ (joblib) so the pipeline and the app can load them."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.SCALER_FILE)
    joblib.dump(classifier, config.CLASSIFIER_FILE)


def load_classifier() -> tuple[StandardScaler, KNeighborsClassifier]:
    """Load the saved scaler and k-NN. Outputs: (scaler, classifier)."""
    return joblib.load(config.SCALER_FILE), joblib.load(config.CLASSIFIER_FILE)


def predict_kidney_probability(scaler: StandardScaler, classifier: KNeighborsClassifier,
                               features: pd.DataFrame) -> np.ndarray:
    """Standardise with the TRAINING scaler, then return P(kidney) for each candidate.

    With uniform weights this is the fraction of the k neighbours that are kidney;
    a candidate is classified kidney when it is >= 0.5.

    Inputs:  fitted scaler, fitted k-NN, feature table (may hold all 17 columns; only the
             columns the scaler was trained on are used, in the same order).
    Outputs: 1D array of probabilities (empty if there are no candidates).
    """
    if len(features) == 0:
        return np.zeros(0)
    kidney_column = list(classifier.classes_).index(1)
    used_columns = features[list(scaler.feature_names_in_)]
    return classifier.predict_proba(scaler.transform(used_columns))[:, kidney_column]


# =============================================================================
# 4. Using the classifier inside the segmentation
# =============================================================================
def blobs_on_side(blobs: list[np.ndarray], anatomy: dict, side: str) -> list[np.ndarray]:
    """Keep only the blobs whose centroid lies on the given side of the spine."""
    return [blob for blob in blobs if phase2_segmentation.blob_position(blob, anatomy)["side"] == side]


def best_kidney_blob(blobs: list[np.ndarray], probabilities: np.ndarray, anatomy: dict,
                     priors: dict) -> tuple[np.ndarray | None, float]:
    """The blob with the highest P(kidney), if that probability is at least 0.5; else (None, best p).

    Ties are common: with k = 7 neighbours P(kidney) can only be 0, 1/7, ..., 7/7, so
    several blobs can score 1.0. A tie is broken by the blob closest to the usual
    kidney position (the same ranking the Phase 2 rule filter uses).
    """
    if len(blobs) == 0:
        return None, 0.0
    ranking = sorted(range(len(blobs)), key=lambda i: (
        -probabilities[i], phase2_segmentation.distance_from_usual_kidney_position(blobs[i], anatomy, priors)))
    best = ranking[0]
    if probabilities[best] < 0.5:
        return None, float(probabilities[best])
    return blobs[best], float(probabilities[best])


def grow_side_with_retries(enhanced_image: np.ndarray, hu_slice: np.ndarray, windowed_image: np.ndarray,
                           anatomy: dict, priors: dict, input_name: str, side: str,
                           scaler=None, classifier=None,
                           max_attempts: int = config.SEED_MAX_ATTEMPTS) -> list[dict]:
    """Strategy B on one side: grow from the best seed; if the k-NN rejects the region, try the next seed.

    Attempt 1 uses the usual seed (learned offset, checked for a kidney-like grey
    level). Each rejected region is excluded from later seed searches, so the next
    attempt starts from the nearest kidney-like pixel that is NOT in a rejected
    region. Stops when the k-NN accepts a blob, when no seed is left, or after
    max_attempts.

    Without a classifier (scaler=None) every attempt is run and nothing is accepted:
    that is how the TRAINING candidates are generated.

    Inputs:
        enhanced_image: input for region growing.  hu_slice, windowed_image: for features.
        anatomy, priors, input_name: as in phase2.  side: "patient_right" / "patient_left".
        scaler, classifier: the k-NN (or None).  max_attempts: see config.
    Outputs:
        List of attempts, each {"seed", "status", "region", "blobs", "probabilities", "accepted"}.
    """
    blocked = ~anatomy["body"] | anatomy["bone_exclusion"]
    search_radius = int(round(config.SEED_SEARCH_RADIUS_FRACTION * anatomy["body_width"]))
    max_area_px = int(priors["max_region_area_fraction"] * anatomy["body_area"])
    expected = phase2_segmentation.expected_seed_position(anatomy, priors, side)
    rejected = np.zeros(enhanced_image.shape, dtype=bool)

    attempts = []
    for _ in range(max_attempts):
        seed, status = phase2_segmentation.find_kidney_like_seed(
            enhanced_image, expected, priors["intensity_range"][input_name], blocked | rejected, search_radius)
        if seed is None:
            break
        region = phase2_segmentation.seeded_region_growing(enhanced_image, seed, config.REGION_GROW_TOLERANCE,
                                                           max_area_px, blocked)
        cleaned = phase2_segmentation.clean_candidates(region, anatomy,
                                                       config.OPENING_RADIUS_FOR_STRATEGY["region_growing"])
        blobs = blobs_on_side(extract_candidate_blobs(cleaned), anatomy, side)
        attempt = {"seed": seed, "status": status, "region": region, "blobs": blobs,
                   "probabilities": np.zeros(len(blobs)), "accepted": None}
        attempts.append(attempt)

        if scaler is not None:
            attempt["probabilities"] = predict_kidney_probability(
                scaler, classifier, features_for_blobs(blobs, hu_slice, windowed_image, anatomy))
            attempt["accepted"], _ = best_kidney_blob(blobs, attempt["probabilities"], anatomy, priors)
            if attempt["accepted"] is not None:
                break
        # Rejected (or no classifier): never seed inside this region again.
        rejected |= region
    return attempts


def multi_otsu_candidate_blobs(enhanced_image: np.ndarray, anatomy: dict) -> list[np.ndarray]:
    """Strategy A candidates: every blob of the multi-Otsu mask after morphology."""
    raw = phase2_segmentation.segment_multi_otsu_candidates(enhanced_image, anatomy, config.MULTI_OTSU_CLASSES,
                                                            config.MULTI_OTSU_KIDNEY_CLASSES)
    cleaned = phase2_segmentation.clean_candidates(raw, anatomy, config.OPENING_RADIUS_FOR_STRATEGY["multi_otsu"])
    return extract_candidate_blobs(cleaned)


def kidney_probability_of_regions(final_mask: np.ndarray, hu_slice: np.ndarray, anatomy: dict,
                                  scaler, classifier) -> list[dict]:
    """Run the k-NN on each region of a FINAL mask and report P(kidney) - a check, not a filter.

    Used when the rule filter picks the regions (config.USE_CLASSIFIER_IN_PIPELINE False):
    the classifier still looks at every region the pipeline returns, so the app can
    show how kidney-like each one is, but it does not delete anything.

    Outputs:
        One dict per region: {"side" ("right"/"left" = patient's), "area_px", "p_kidney", "mask"}.
    """
    windowed = phase1_enhancement.apply_hu_window(hu_slice)
    regions = extract_candidate_blobs(final_mask, min_area_px=1)
    probabilities = predict_kidney_probability(scaler, classifier,
                                               features_for_blobs(regions, hu_slice, windowed, anatomy))
    side_names = {"patient_right": "right", "patient_left": "left"}
    return [{"side": side_names[phase2_segmentation.blob_position(region, anatomy)["side"]],
             "area_px": int(region.sum()), "p_kidney": float(probability), "mask": region}
            for region, probability in zip(regions, probabilities)]


def segment_with_knn(hu_slice: np.ndarray, enhanced_image: np.ndarray, anatomy: dict, priors: dict,
                     strategy: str, input_name: str, scaler, classifier,
                     max_attempts: int = config.SEED_MAX_ATTEMPTS) -> dict:
    """Segment one slice with the k-NN deciding which candidates are kidney.

    Strategy A ("multi_otsu"): classify every candidate blob; on each side of the
        spine keep the blob with the highest P(kidney) if it is >= 0.5.
    Strategy B ("region_growing"): per side, grow_side_with_retries.

    Inputs:
        hu_slice: raw HU.  enhanced_image: that strategy's Phase 1 input.
        anatomy, priors: from phase2.  strategy, input_name: as in phase2.
        scaler, classifier: the trained k-NN.  max_attempts: seeds tried per side (B only).
    Outputs:
        dict with "final" (kidney mask), "blobs" and "probabilities" (A), "attempts" per side (B).
    """
    windowed = phase1_enhancement.apply_hu_window(hu_slice)
    final = np.zeros(hu_slice.shape, dtype=bool)

    if strategy == "multi_otsu":
        blobs = multi_otsu_candidate_blobs(enhanced_image, anatomy)
        probabilities = predict_kidney_probability(scaler, classifier,
                                                   features_for_blobs(blobs, hu_slice, windowed, anatomy))
        for side in phase2_segmentation.SIDES:
            side_index = [i for i, blob in enumerate(blobs)
                          if phase2_segmentation.blob_position(blob, anatomy)["side"] == side]
            kept, _ = best_kidney_blob([blobs[i] for i in side_index], probabilities[side_index], anatomy, priors)
            if kept is not None:
                final |= kept
        return {"final": final, "blobs": blobs, "probabilities": probabilities}

    attempts_per_side = {}
    for side in phase2_segmentation.SIDES:
        attempts = grow_side_with_retries(enhanced_image, hu_slice, windowed, anatomy, priors, input_name, side,
                                          scaler, classifier, max_attempts)
        attempts_per_side[side] = attempts
        if attempts and attempts[-1]["accepted"] is not None:
            final |= attempts[-1]["accepted"]
    return {"final": final, "attempts": attempts_per_side}
