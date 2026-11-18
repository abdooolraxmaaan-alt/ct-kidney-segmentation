# Sample test images

The 24 TEST slices used for the final evaluation (KiTS23 cases
13, 16, 20, 22; 6 slices each), included because
the results can be checked without downloading the full dataset.

- `case_XXXXX_slice_YYY.npz` - raw data: `hu` (int16 Hounsfield Units, not windowed),
  `kidney_mask` (1 = kidney, KiTS labels 1+2+3), `label_map` (KiTS labels 0-3),
  `pixel_spacing` (row_mm, col_mm), `case_id`, `slice_index`. Load with `numpy.load`.
- `case_XXXXX_slice_YYY.png` - preview only: soft-tissue window
  [-50, 250] HU
  mapped to 0-255. Use the `.npz` for processing (the PNG has lost the HU values).

Orientation: standard radiology view (patient's front at the top, patient's right on the image left).

## Source and licence

Data from the **KiTS23** challenge (2023 Kidney and Kidney Tumor Segmentation Challenge),
https://kits-challenge.org/kits23/ - Heller, N., Isensee, F., Trofimova, D., et al. (2023).
The KiTS21 challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in
corticomedullary-phase CT. arXiv:2307.01984.

The KiTS23 data is licensed under **CC BY-NC-SA 4.0**
(https://creativecommons.org/licenses/by-nc-sa/4.0/): attribution, non-commercial use
only, and derivatives shared under the same licence. These slices are extracted from the
original volumes without modification of their values.
