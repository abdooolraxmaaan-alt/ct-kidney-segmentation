"""
pipeline.py - the entry point for other programs (and the demo app).

    from src.pipeline import segment_kidney
    mask = segment_kidney(hu_slice)

segment_kidney's name and signature are the public interface; everything inside
can change as the methods improve.
Requires models/segmentation_priors.json (step 3) and, when the k-NN is used,
models/feature_scaler.joblib + models/knn_classifier.joblib (step 4).
"""

import numpy as np

import config
from src import phase1_enhancement, phase2_segmentation, phase3_classification

_models_cache = {}


def _load_models_once() -> dict:
    """Read the learned priors and the k-NN from disk the first time, then reuse them."""
    if not _models_cache:
        _models_cache["priors"] = phase2_segmentation.load_priors()
        _models_cache["scaler"], _models_cache["classifier"] = phase3_classification.load_classifier()
    return _models_cache


def segment_kidney_with_details(hu_slice: np.ndarray) -> dict:
    """Like segment_kidney, but also returns each final region with the k-NN's P(kidney).

    Whichever method picks the regions (k-NN or rule filter, config.USE_CLASSIFIER_IN_PIPELINE),
    the k-NN is run on every final region as a check, so the app can display how
    kidney-like each region looks. It never deletes a region here.

    Inputs:
        hu_slice: 2D raw HU slice.
    Outputs:
        {"mask": 2D boolean kidney mask,
         "regions": [{"side": "right"/"left" (patient's), "area_px", "p_kidney", "mask"}, ...]}
    """
    models = _load_models_once()
    enhanced = phase1_enhancement.enhance_slice(hu_slice)
    anatomy = phase2_segmentation.find_anatomy(hu_slice)
    if config.USE_CLASSIFIER_IN_PIPELINE:
        final = phase3_classification.segment_with_knn(hu_slice, enhanced, anatomy, models["priors"],
                                                       config.FINAL_SEGMENTATION_STRATEGY, config.SEGMENTATION_CONTRAST,
                                                       models["scaler"], models["classifier"])["final"]
    else:
        final = phase2_segmentation.segment_slice(enhanced, anatomy, models["priors"],
                                                  config.FINAL_SEGMENTATION_STRATEGY, config.SEGMENTATION_CONTRAST)["final"]
    regions = phase3_classification.kidney_probability_of_regions(final, hu_slice, anatomy,
                                                                  models["scaler"], models["classifier"])
    return {"mask": final, "regions": regions}


def segment_kidney(hu_slice: np.ndarray) -> np.ndarray:
    """Full pipeline for one CT slice: raw HU in, kidney mask out.

    Steps:
        1. phase1_enhancement.enhance_slice   window -> diffusion -> homomorphic -> contrast step
        2. phase2_segmentation.find_anatomy   body, bone and spine from raw HU
        3. strategy from config.FINAL_SEGMENTATION_STRATEGY, then morphology -> candidates
        4. keep the kidney candidates:
             config.USE_CLASSIFIER_IN_PIPELINE True  -> phase3_classification.segment_with_knn
                 (k-NN picks the blobs; region growing retries the next seed if rejected)
             False -> phase2 rule filter (one blob per side inside the learned zone)
        5. the k-NN reports P(kidney) for each final region (see segment_kidney_with_details)

    Inputs:
        hu_slice: 2D numpy array of raw Hounsfield Units (one axial KiTS23 slice,
                  standard orientation - see README "Slice file format").
    Outputs:
        2D boolean mask, same shape as hu_slice, True = kidney (both kidneys).
    """
    return segment_kidney_with_details(hu_slice)["mask"]


def plot_contour_overlay(ax, display_image: np.ndarray, predicted_mask: np.ndarray,
                         ground_truth_mask: np.ndarray | None = None, title: str = "") -> None:
    """Draw the CT slice with the predicted outline in red and the ground truth in green.

    Shared by the scripts (example and failure-case figures) and usable by other programs.

    Inputs:
        ax: a matplotlib Axes to draw on.
        display_image: 2D uint8 windowed slice.
        predicted_mask: 2D boolean mask (red contour).
        ground_truth_mask: optional 2D boolean mask (green contour).
        title: text above the image (e.g. case, slice and Dice).
    Outputs:
        None (draws on ax).
    """
    ax.imshow(display_image, cmap="gray")
    if ground_truth_mask is not None and ground_truth_mask.any():
        ax.contour(ground_truth_mask, levels=[0.5], colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=1.0)
    if predicted_mask.any():
        ax.contour(predicted_mask, levels=[0.5], colors=config.PREDICTED_CONTOUR_COLOUR, linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
