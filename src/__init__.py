"""Kidney segmentation for abdominal CT (KiTS23) with classical image processing.

One module per phase:
    data_loading            - download KiTS23 cases, pick 6 slices per patient, train/test split
    phase1_enhancement      - HU window, filtering, contrast, frequency domain
    phase2_segmentation     - multi-Otsu, seeded region growing, morphology
    phase3_classification   - blob features (17) and k-NN classifier
    metrics                 - PSNR, CNR, Dice, IoU, classification scores
    pipeline                - segment_kidney(hu_slice) -> mask, the entry point for other programs
"""
