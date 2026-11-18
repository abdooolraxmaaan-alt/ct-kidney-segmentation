"""
Step 1 - prepare the data.

Run from the project folder:  .venv/bin/python scripts/step1_prepare_data.py

What it does:
    1. downloads KiTS23 cases 12-23 into data/raw/ (skips files already there)
    2. for each patient: finds the slices that contain kidney (labels 1+2+3) and
       picks 6 evenly spaced slices, skipping the outer 10% at the top and bottom
    3. saves each slice to data/slices/case_000XX_slice_YYY.npz (format in README.md)
    4. splits the patients 8 train / 4 test (seed 42) and saves data/split.json

Outputs:
    results/figures/step1_all_slices_overview.png - all 72 slices, green = ground-truth kidney
    results/tables/step1_dataset_summary.csv      - one row per slice

Safe to re-run: downloads are skipped and slice files are simply rewritten.
"""

import sys
from pathlib import Path

# Let this script import config.py and src/ from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to file without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_loading  # noqa: E402
from src.phase1_enhancement import apply_hu_window  # noqa: E402


def extract_and_save_patient_slices(case_id: int) -> list[dict]:
    """Load one patient's volume, choose 6 kidney slices, save them, and describe each one.

    Inputs:
        case_id: KiTS23 case number.
    Outputs:
        List of 6 summary rows (dicts) for the dataset summary table.
    """
    hu_volume, label_volume, pixel_spacing = data_loading.load_volume(case_id)
    kidney_volume = data_loading.make_whole_kidney_mask(label_volume)
    first_slice, last_slice = data_loading.find_kidney_slice_range(kidney_volume)
    chosen_slices = data_loading.choose_slice_indices(first_slice, last_slice)

    print(f"    volume {hu_volume.shape}, pixel {pixel_spacing[0]:.2f} x {pixel_spacing[1]:.2f} mm, "
          f"HU {hu_volume.min():.0f}..{hu_volume.max():.0f}")
    print(f"    kidney on slices {first_slice}-{last_slice} -> chosen {chosen_slices}")

    rows = []
    for slice_index in chosen_slices:
        label_slice = label_volume[slice_index]
        data_loading.save_slice(case_id, slice_index, hu_volume[slice_index], label_slice, pixel_spacing)

        kidney_pixels = int(kidney_volume[slice_index].sum())
        if kidney_pixels == 0:
            # Possible if the two kidneys do not overlap in height and a slice falls in the gap.
            print(f"    WARNING: slice {slice_index} has no kidney pixels")
        rows.append({
            "case_id": case_id,
            "slice_index": slice_index,
            "kidney_pixels": kidney_pixels,
            "kidney_area_mm2": round(kidney_pixels * pixel_spacing[0] * pixel_spacing[1], 1),
            "tumour_present": bool((label_slice == config.TUMOUR_LABEL).any()),
            "cyst_present": bool((label_slice == config.CYST_LABEL).any()),
            "kidney_first_slice": first_slice,
            "kidney_last_slice": last_slice,
        })
    return rows


def save_overview_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """Draw all slices in a grid: one row per patient, top-to-bottom of the kidney left-to-right.

    Each panel shows the soft-tissue window with the ground-truth kidney outlined in
    green, titled with case, slice number and train/test.

    Inputs:
        summary: the dataset summary table (one row per slice, with a "split" column).
        output_path: where to save the PNG.
    Outputs:
        None (writes the figure).
    """
    case_ids = sorted(summary["case_id"].unique())
    n_columns = config.SLICES_PER_PATIENT
    figure, axes = plt.subplots(len(case_ids), n_columns,
                                figsize=(n_columns * 2.4, len(case_ids) * 2.6))

    for row_number, case_id in enumerate(case_ids):
        patient_rows = summary[summary["case_id"] == case_id].sort_values("slice_index")
        for column_number, (_, row) in enumerate(patient_rows.iterrows()):
            axis = axes[row_number, column_number]
            slice_data = data_loading.load_slice(
                data_loading.slice_file_path(row["case_id"], row["slice_index"]))

            axis.imshow(apply_hu_window(slice_data["hu"]), cmap="gray")
            if slice_data["ground_truth"].any():
                axis.contour(slice_data["ground_truth"], levels=[0.5],
                             colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=0.8)

            title_colour = "tab:blue" if row["split"] == "train" else "tab:red"
            axis.set_title(f"case {case_id} | slice {row['slice_index']} | {row['split']}",
                           fontsize=7, color=title_colour)
            axis.axis("off")

    figure.suptitle("All 72 slices (soft-tissue window W400/L40). Green = ground-truth kidney (labels 1+2+3).\n"
                    "Each row = one patient, kidney top (left) to bottom (right). Blue title = train, red = test.",
                    fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=config.FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    # 1. Download (skips what is already there).
    print("1) Downloading KiTS23 cases", config.CASE_IDS[0], "to", config.CASE_IDS[-1])
    for case_id in config.CASE_IDS:
        print(f"  case {case_id:05d}")
        data_loading.download_case(case_id)

    # 2-3. Choose and save 6 slices per patient.
    print("\n2) Choosing and saving slices "
          f"({config.SLICES_PER_PATIENT} per patient, {config.SLICE_EDGE_MARGIN_FRACTION:.0%} margin each end)")
    summary_rows = []
    for case_id in config.CASE_IDS:
        print(f"  case {case_id:05d}")
        summary_rows.extend(extract_and_save_patient_slices(case_id))

    # 4. Split by patient with the fixed seed.
    train_ids, test_ids = data_loading.split_patients_train_test()
    data_loading.save_split(train_ids, test_ids)
    print(f"\n3) Split (seed {config.RANDOM_SEED}) saved to {config.SPLIT_FILE.relative_to(config.PROJECT_ROOT)}")
    print(f"   train ({len(train_ids)} patients, {len(train_ids) * config.SLICES_PER_PATIENT} slices): {train_ids}")
    print(f"   test  ({len(test_ids)} patients, {len(test_ids) * config.SLICES_PER_PATIENT} slices): {test_ids}")

    # 5. Summary table (one row per slice).
    summary = pd.DataFrame(summary_rows)
    summary.insert(2, "split", np.where(summary["case_id"].isin(train_ids), "train", "test"))
    summary["file"] = [config.SLICE_FILE_NAME.format(case=c, slice=s)
                       for c, s in zip(summary["case_id"], summary["slice_index"])]
    config.DATASET_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.DATASET_SUMMARY_CSV, index=False)
    print(f"\n4) Summary table: {config.DATASET_SUMMARY_CSV.relative_to(config.PROJECT_ROOT)} ({len(summary)} rows)")

    # 6. Overview figure.
    overview_path = config.FIGURES_DIR / "step1_all_slices_overview.png"
    save_overview_figure(summary, overview_path)
    print(f"5) Overview figure: {overview_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
