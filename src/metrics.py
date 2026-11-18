"""
metrics.py - every number used to judge the pipeline.

    Phase 1: psnr, cnr (+ make_background_ring to define CNR's background),
             edge_strength_on_boundary (does a filter blur the kidney edge?)
    Phase 2: dice_coefficient, iou_score
    Phase 3: binary_classification_scores, binary_confusion_matrix

Textbook references ("Textbook: Ch. N") are to Gonzalez, R. C., & Woods, R. E. (2018).
Digital image processing (4th ed., US edition). Pearson.
"""

import numpy as np
from scipy import ndimage


def psnr(reference: np.ndarray, test: np.ndarray, max_value: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio between a reference image and a processed image.

    PSNR = 10 * log10(MAX^2 / MSE), where MSE is the mean squared pixel difference.

    IMPORTANT - what PSNR does and does NOT tell us here: we have no noise-free
    "clean" CT to compare against, so the reference is the windowed (unfiltered)
    slice. PSNR therefore measures how CLOSE the output stays to the input, not how
    well noise was removed. A filter that does nothing scores infinity; a filter that
    removes a lot of noise necessarily scores lower. Read it together with CNR and
    the edge-strength ratio, never on its own.

    Inputs:
        reference, test: 2D arrays of the same shape (uint8 or float).
        max_value: largest possible pixel value (255 for 8-bit).
    Outputs:
        PSNR in decibels (float('inf') if the images are identical).
    Reference:
        Huynh-Thu, Q., & Ghanbari, M. (2008). Scope of validity of PSNR in image/video
        quality assessment. Electronics Letters, 44(13), 800-801.
    """
    # float64 so squaring uint8 differences cannot overflow.
    difference = reference.astype(np.float64) - test.astype(np.float64)
    mean_squared_error = np.mean(difference ** 2)
    if mean_squared_error == 0:
        return float("inf")
    return float(10.0 * np.log10(max_value ** 2 / mean_squared_error))


def make_background_ring(kidney_mask: np.ndarray, ring_width: int) -> np.ndarray:
    """Build a ring of pixels just outside the kidney, used as CNR's "background".

    Every pixel whose straight-line distance to the nearest kidney pixel is between
    1 and ring_width belongs to the ring. This is the same as dilating the kidney
    with a disk of radius ring_width and removing the kidney itself.

    Inputs:
        kidney_mask: 2D boolean ground-truth kidney.
        ring_width:  ring thickness in pixels (config.CNR_BACKGROUND_RING_PX).
    Outputs:
        2D boolean ring mask.
    Textbook: Ch. 9 (Morphological Image Processing) - dilation with a disk.
    """
    # Distance from every non-kidney pixel to the nearest kidney pixel (0 inside the kidney).
    distance_to_kidney = ndimage.distance_transform_edt(~kidney_mask)
    return (distance_to_kidney > 0) & (distance_to_kidney <= ring_width)


def cnr(image: np.ndarray, tissue_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """Contrast-to-Noise Ratio between kidney tissue and the surrounding background.

    CNR = |mean(kidney) - mean(background)| / std(background).
    Higher = the kidney stands out more clearly from its surroundings relative to noise.

    Two properties worth knowing when reading the results:
      * A purely linear change of brightness/contrast (a*x + b) does not change CNR,
        because the difference and the std scale together. Only non-linear steps
        (clipping, equalisation, filtering) can change it.
      * std(background) includes real anatomy in the ring (fat next to muscle or
        liver), not only noise - so CNR also rewards smoothing that blurs anatomy.
        That is why we also check edge strength.

    Inputs:
        image: 2D image (after a given enhancement step).
        tissue_mask, background_mask: 2D boolean masks.
    Outputs:
        CNR as a float (0.0 if the background has no variation at all).
    Reference:
        Bushberg, J. T., Seibert, J. A., Leidholdt, E. M., & Boone, J. M. (2012).
        The essential physics of medical imaging (3rd ed.). Lippincott Williams & Wilkins.
    """
    pixels = image.astype(np.float64)
    tissue_mean = pixels[tissue_mask].mean()
    background_mean = pixels[background_mask].mean()
    background_std = pixels[background_mask].std()
    if background_std == 0:
        return 0.0  # avoid dividing by zero on a perfectly flat background
    return float(abs(tissue_mean - background_mean) / background_std)


def edge_strength_on_boundary(image: np.ndarray, kidney_mask: np.ndarray) -> float:
    """Average gradient magnitude (Sobel) on the ground-truth kidney boundary.

    Used as a ratio: edge_strength(filtered) / edge_strength(windowed input).
    1.0 = the kidney edge is as sharp as before; below 1 = the filter blurred it.
    This turns the goal "remove noise without blurring organ edges" into a number.

    Inputs:
        image: 2D image.
        kidney_mask: 2D boolean ground-truth kidney.
    Outputs:
        Mean Sobel gradient magnitude over the kidney's boundary pixels.
    Textbook: Ch. 3 (Intensity Transformations and Spatial Filtering) - Sobel gradient; Ch. 10 (Image Segmentation) - edge detection.
    """
    pixels = image.astype(np.float64)
    gradient_rows = ndimage.sobel(pixels, axis=0)
    gradient_cols = ndimage.sobel(pixels, axis=1)
    gradient_magnitude = np.hypot(gradient_rows, gradient_cols)

    # Boundary = kidney pixels that have at least one non-kidney neighbour.
    boundary = kidney_mask & ~ndimage.binary_erosion(kidney_mask)
    return float(gradient_magnitude[boundary].mean())


def dice_coefficient(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Dice Similarity Coefficient: Dice = 2|A n B| / (|A| + |B|).

    Inputs:
        predicted_mask, ground_truth_mask: 2D boolean masks.
    Outputs:
        Dice in [0, 1]; 1 = perfect overlap. Returns 1.0 if both masks are empty.
    Reference:
        Dice, L. R. (1945). Measures of the amount of ecologic association between
        species. Ecology, 26(3), 297-302.
        Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image
        segmentation: analysis, selection, and tool. BMC Medical Imaging, 15, 29.
    """
    predicted, truth = predicted_mask.astype(bool), ground_truth_mask.astype(bool)
    total = predicted.sum() + truth.sum()
    if total == 0:
        return 1.0  # both empty: perfect agreement that there is nothing
    overlap = np.logical_and(predicted, truth).sum()
    return float(2.0 * overlap / total)


def iou_score(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Jaccard index / Intersection over Union: IoU = |A n B| / |A u B|.

    Inputs:
        predicted_mask, ground_truth_mask: 2D boolean masks.
    Outputs:
        IoU in [0, 1]. Returns 1.0 if both masks are empty.
    Reference:
        Jaccard, P. (1912). The distribution of the flora in the alpine zone.
        New Phytologist, 11(2), 37-50.
    """
    predicted, truth = predicted_mask.astype(bool), ground_truth_mask.astype(bool)
    union = np.logical_or(predicted, truth).sum()
    if union == 0:
        return 1.0
    overlap = np.logical_and(predicted, truth).sum()
    return float(overlap / union)


def binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix for kidney (1) vs not-kidney (0).

    Inputs:
        y_true, y_pred: 1D arrays of 0/1 labels.
    Outputs:
        [[TN, FP], [FN, TP]] as a 2x2 int array.
    Textbook: Ch. 13 (Image Pattern Classification) - evaluating a classifier.
    """
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])


def binary_classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, recall (sensitivity), specificity, precision and F1 from the confusion matrix.

    Inputs:
        y_true, y_pred: 1D arrays of 0/1 labels.
    Outputs:
        dict with "accuracy", "recall" (= sensitivity), "specificity", "precision", "f1"
        and the four confusion-matrix counts.
        accuracy = (TP+TN)/all; recall = TP/(TP+FN); specificity = TN/(TN+FP);
        precision = TP/(TP+FP); F1 = 2*precision*recall/(precision+recall).
    Textbook: Ch. 13 (Image Pattern Classification).
    """
    (true_negatives, false_positives), (false_negatives, true_positives) = binary_confusion_matrix(y_true, y_pred)

    def safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else 0.0

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)          # sensitivity
    return {
        "accuracy": safe_divide(true_positives + true_negatives,
                                true_positives + true_negatives + false_positives + false_negatives),
        "recall": recall,
        "specificity": safe_divide(true_negatives, true_negatives + false_positives),
        "precision": precision,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "true_positives": true_positives, "false_positives": false_positives,
        "true_negatives": true_negatives, "false_negatives": false_negatives,
    }
