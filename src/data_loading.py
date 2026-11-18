"""
data_loading.py - getting KiTS23 data into the shape the pipeline needs.

Steps (all called from scripts/step1_prepare_data.py):
    1. download my 12 patients (cases 12-23)                   -> download_case
    2. load each 3D CT volume (in Hounsfield Units) and its
       label volume, turned to the standard viewing orientation -> load_volume
    3. build the whole-kidney ground truth (labels 1 + 2 + 3)    -> make_whole_kidney_mask
    4. pick 6 axial slices per patient between the kidney's
       top and bottom, skipping a 10% margin at each end         -> find_kidney_slice_range, choose_slice_indices
    5. save each slice so later steps never reopen the volumes   -> save_slice, load_slice
    6. split the patients 8 train / 4 test with a fixed seed     -> split_patients_train_test, save_split

Later steps only need load_split("train") / load_split("test").

Dataset reference:
    Heller, N., Isensee, F., Trofimova, D., et al. (2023). The KiTS21 challenge:
    Automatic segmentation of kidneys, renal tumors, and renal cysts in
    corticomedullary-phase CT. arXiv:2307.01984.
"""

import json
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

import config


# -----------------------------------------------------------------------------
# 1. Download
# -----------------------------------------------------------------------------
def download_file_with_progress(url: str, destination: Path) -> None:
    """Download one file, printing a percentage as it goes.

    The file is first written as "<name>.partial" and only renamed when complete,
    so an interrupted download is never mistaken for a finished one.

    Inputs:
        url: web address of the file.
        destination: where to save it.
    Outputs:
        None (the file appears at destination).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".partial")

    def report_progress(blocks_done: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100.0, 100.0 * blocks_done * block_size / total_size)
            print(f"\r    {destination.name}: {percent:5.1f}% of {total_size / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, partial_path, reporthook=report_progress)
    print()  # end the progress line
    partial_path.rename(destination)


def case_folder(case_id: int) -> Path:
    """Folder that holds one patient's raw files, e.g. data/raw/case_00012."""
    return config.RAW_DIR / f"case_{case_id:05d}"


def download_case(case_id: int) -> tuple[Path, Path]:
    """Download one KiTS23 patient (CT volume + label volume) unless already on disk.

    Inputs:
        case_id: KiTS23 case number, e.g. 12 for case_00012.
    Outputs:
        (image_path, mask_path) - paths to imaging.nii.gz and segmentation.nii.gz
        under data/raw/case_000XX/.
    Reference:
        KiTS23 dataset (Heller et al., 2023). Same sources the official kits23
        repository uses: images on Hugging Face, masks on GitHub (see config).
    """
    image_path = case_folder(case_id) / "imaging.nii.gz"
    mask_path = case_folder(case_id) / "segmentation.nii.gz"

    for url_template, path in [(config.KITS23_IMAGE_URL, image_path), (config.KITS23_MASK_URL, mask_path)]:
        if path.exists():
            print(f"    {path.name}: already downloaded, skipping")
        else:
            download_file_with_progress(url_template.format(case=case_id), path)
    return image_path, mask_path


# -----------------------------------------------------------------------------
# 2. Load and orient
# -----------------------------------------------------------------------------
def reorient_to_display_orientation(nifti_image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Turn a volume so slices display the standard radiology way.

    Target (config.DISPLAY_ORIENTATION = I, P, L): slice 0 at the top of the body,
    patient's front at the top of each image, patient's right on the image's left.
    The file's affine matrix says how its axes point; nibabel works out the flips
    and axis swaps needed. For KiTS23 this changes nothing (files are already stored
    this way), but the check protects us if a case is stored differently.

    Inputs:
        nifti_image: image loaded with nibabel.
    Outputs:
        The same image with its axes reordered/flipped (affine updated to match).
    """
    current_orientation = nib.orientations.io_orientation(nifti_image.affine)
    target_orientation = nib.orientations.axcodes2ornt(config.DISPLAY_ORIENTATION)
    transform = nib.orientations.ornt_transform(current_orientation, target_orientation)
    return nifti_image.as_reoriented(transform)


def load_volume(case_id: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Load one patient's CT volume (HU) and label volume, in display orientation.

    Raw HU are kept (not 0-255) at this stage; windowing happens in Phase 1.

    Inputs:
        case_id: KiTS23 case number.
    Outputs:
        hu_volume:     3D float32 array (slices, rows, cols) of raw Hounsfield Units.
        label_volume:  3D uint8 array, 0 background / 1 kidney / 2 tumour / 3 cyst.
        pixel_spacing: (row_mm, col_mm) - real size of one pixel, needed to turn
                       pixel areas into mm^2.
    """
    image_path = case_folder(case_id) / "imaging.nii.gz"
    mask_path = case_folder(case_id) / "segmentation.nii.gz"

    ct_image = reorient_to_display_orientation(nib.load(image_path))
    label_image = reorient_to_display_orientation(nib.load(mask_path))

    # float32 halves memory compared with nibabel's default float64; HU are whole numbers anyway.
    hu_volume = ct_image.get_fdata(dtype=np.float32)
    label_volume = np.asarray(label_image.dataobj).astype(np.uint8)

    # Sanity checks: a mismatch here would silently pair the wrong mask with the CT.
    if hu_volume.shape != label_volume.shape:
        raise ValueError(f"case {case_id}: CT shape {hu_volume.shape} != mask shape {label_volume.shape}")
    unexpected_labels = set(np.unique(label_volume)) - {0, 1, 2, 3}
    if unexpected_labels:
        raise ValueError(f"case {case_id}: unexpected label values {unexpected_labels}")

    zooms = ct_image.header.get_zooms()          # (slice thickness, row spacing, col spacing) in mm
    pixel_spacing = (float(zooms[1]), float(zooms[2]))
    return hu_volume, label_volume, pixel_spacing


# -----------------------------------------------------------------------------
# 3-4. Ground truth and slice choice
# -----------------------------------------------------------------------------
def make_whole_kidney_mask(label_map: np.ndarray) -> np.ndarray:
    """Turn the KiTS23 label map into a single yes/no "whole kidney" mask.

    This is the ground truth that Dice and IoU compare against. Tumour and cyst are
    included because they lie inside the kidney outline.

    Inputs:
        label_map: label array (2D slice or 3D volume) with values 0-3.
    Outputs:
        Boolean array, True where the label is in config.KIDNEY_LABELS (1, 2, 3).
    """
    return np.isin(label_map, config.KIDNEY_LABELS)


def find_kidney_slice_range(kidney_mask_volume: np.ndarray) -> tuple[int, int]:
    """Find the first and last axial slice that contain any kidney.

    Inputs:
        kidney_mask_volume: 3D boolean whole-kidney mask (slices first).
    Outputs:
        (first_slice, last_slice) - inclusive; first = top of the kidneys, last = bottom.
    """
    slice_has_kidney = kidney_mask_volume.any(axis=(1, 2))
    kidney_slices = np.flatnonzero(slice_has_kidney)
    if kidney_slices.size == 0:
        raise ValueError("no kidney found in this volume")
    return int(kidney_slices[0]), int(kidney_slices[-1])


def choose_slice_indices(first_slice: int, last_slice: int,
                         n_slices: int = config.SLICES_PER_PATIENT,
                         edge_margin_fraction: float = config.SLICE_EDGE_MARGIN_FRACTION) -> list[int]:
    """Pick n evenly spaced slice numbers between the kidney's top and bottom margins.

    Example: kidney on slices 11-39 (extent 28), margin 0.1 -> skip round(2.8) = 3
    slices at each end -> spread 6 slices evenly over 14-36.

    Inputs:
        first_slice, last_slice: the kidney's slice range (from find_kidney_slice_range).
        n_slices: how many slices to pick (config.SLICES_PER_PATIENT, 6).
        edge_margin_fraction: fraction of the extent skipped at each end (see config).
    Outputs:
        Sorted list of n_slices distinct slice numbers.
    """
    extent = last_slice - first_slice
    margin = int(round(edge_margin_fraction * extent))
    start, end = first_slice + margin, last_slice - margin

    # Evenly spaced so every part of the kidney (upper pole, hilum, lower pole) is represented.
    chosen = [int(round(z)) for z in np.linspace(start, end, n_slices)]
    if len(set(chosen)) != n_slices:
        raise ValueError(f"kidney range {first_slice}-{last_slice} is too short for {n_slices} distinct slices")
    return chosen


# -----------------------------------------------------------------------------
# 5. Save / load single slices
# -----------------------------------------------------------------------------
def slice_file_path(case_id: int, slice_index: int) -> Path:
    """Path of one saved slice, e.g. data/slices/case_00012_slice_045.npz."""
    return config.SLICES_DIR / config.SLICE_FILE_NAME.format(case=case_id, slice=slice_index)


def save_slice(case_id: int, slice_index: int, hu_slice: np.ndarray, label_slice: np.ndarray,
               pixel_spacing: tuple[float, float]) -> Path:
    """Save one 2D slice (raw HU + ground truth + metadata) as a compressed .npz.

    Contents (format also documented in README.md):
        hu            int16 (rows, cols)  raw Hounsfield Units, NOT windowed
        kidney_mask   uint8 (rows, cols)  1 = whole kidney (labels 1+2+3), 0 = not
        label_map     uint8 (rows, cols)  original KiTS23 labels 0-3
        pixel_spacing float32 (2,)        (row_mm, col_mm)
        case_id       int                 e.g. 12
        slice_index   int                 axial slice number in the original volume

    Inputs:
        case_id, slice_index: identify the slice.
        hu_slice: 2D HU array.  label_slice: 2D label array (0-3).
        pixel_spacing: (row_mm, col_mm).
    Outputs:
        Path of the saved file.
    """
    # KiTS23 HU values are whole numbers between about -1024 and 3071, so int16 is exact
    # and a quarter of the size of float64. Check rather than assume.
    if not np.array_equal(hu_slice, np.round(hu_slice)):
        raise ValueError(f"case {case_id} slice {slice_index}: HU values are not whole numbers")

    path = slice_file_path(case_id, slice_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hu=hu_slice.astype(np.int16),
        kidney_mask=make_whole_kidney_mask(label_slice).astype(np.uint8),
        label_map=label_slice.astype(np.uint8),
        pixel_spacing=np.array(pixel_spacing, dtype=np.float32),
        case_id=np.int64(case_id),
        slice_index=np.int64(slice_index),
    )
    return path


def load_slice(slice_path: Path) -> dict:
    """Load one saved slice.

    Inputs:
        slice_path: path to a .npz written by save_slice.
    Outputs:
        dict with keys "hu" (float32 HU), "ground_truth" (bool kidney mask),
        "label_map", "pixel_spacing", "case_id", "slice_index".
    """
    with np.load(slice_path) as data:
        return {
            "hu": data["hu"].astype(np.float32),
            "ground_truth": data["kidney_mask"].astype(bool),
            "label_map": data["label_map"],
            "pixel_spacing": tuple(float(v) for v in data["pixel_spacing"]),
            "case_id": int(data["case_id"]),
            "slice_index": int(data["slice_index"]),
        }


# -----------------------------------------------------------------------------
# 6. Train / test split by patient
# -----------------------------------------------------------------------------
def split_patients_train_test(case_ids: list[int] = config.CASE_IDS,
                              n_train: int = config.N_TRAIN_PATIENTS,
                              seed: int = config.RANDOM_SEED) -> tuple[list[int], list[int]]:
    """Split the patients (not the slices) into training and testing groups.

    Splitting by patient stops slices of the same patient appearing in both sets,
    which would inflate the results.

    Inputs:
        case_ids: all my patient numbers (12-23).
        n_train:  how many go to training (8 -> 8/4 split, about 70/30).
        seed:     fixed random seed so the split is the same every run.
    Outputs:
        (train_case_ids, test_case_ids), each sorted.
    """
    random_generator = np.random.default_rng(seed)
    shuffled = random_generator.permutation(case_ids)
    train_ids = sorted(int(c) for c in shuffled[:n_train])
    test_ids = sorted(int(c) for c in shuffled[n_train:])
    return train_ids, test_ids


def save_split(train_ids: list[int], test_ids: list[int]) -> None:
    """Write the split to config.SPLIT_FILE (data/split.json) so every step uses the same one."""
    split = {"seed": config.RANDOM_SEED, "train": train_ids, "test": test_ids}
    config.SPLIT_FILE.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split_case_ids() -> dict:
    """Read data/split.json. Outputs: {"seed": 42, "train": [...], "test": [...]}."""
    return json.loads(config.SPLIT_FILE.read_text(encoding="utf-8"))


def load_split(split: str) -> list[dict]:
    """Load every saved slice of one split, ready for the phase scripts.

    If data/ has not been prepared (e.g. a fresh copy of the repository, where data/
    is not included), the TEST slices are read from sample_test_images/ instead, so
    everything that only needs the test slices still works. The training slices are
    not shipped, so asking for them without data/ raises a clear error.

    Keeps tuning (train) and final evaluation (test) strictly separate.

    Inputs:
        split: "train" or "test".
    Outputs:
        List of slice dicts (see load_slice), sorted by case then slice number.
    """
    if split not in ("train", "test"):
        raise ValueError('split must be "train" or "test"')
    if not config.SPLIT_FILE.exists():
        if split == "test":
            sample_paths = sorted(config.SAMPLE_TEST_IMAGES_DIR.glob("case_*_slice_*.npz"))
            if sample_paths:
                return [load_slice(path) for path in sample_paths]
        raise FileNotFoundError(f"No prepared {split} slices: run scripts/step1_prepare_data.py first "
                                "(it downloads KiTS23 and creates data/).")
    case_ids = load_split_case_ids()[split]

    slices = []
    for case_id in case_ids:
        for path in sorted(config.SLICES_DIR.glob(f"case_{case_id:05d}_slice_*.npz")):
            slices.append(load_slice(path))
    return slices
