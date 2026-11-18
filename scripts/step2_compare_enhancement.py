"""
Step 2 - compare the Phase 1 enhancement methods and choose their settings.

Run from the project folder:  .venv/bin/python scripts/step2_compare_enhancement.py

Every choice is made on the 8 TRAINING patients (48 slices). The stages are tuned
in the same order as the final chain, each on the output of the previous winner:

    1. HU window          3 windows compared by CNR
    2. noise removal      Perona-Malik (kappa x iterations x conduction) vs median (3, 5, 7)
                          winner = highest CNR that keeps >= 90% of the kidney edge strength
    3. homomorphic filter gamma_low x gamma_high x D0, winner = highest CNR
    4. contrast           CLAHE (clip limit x tile grid) vs global histogram equalisation

The settings found here must then be copied into config.py (the script prints
them and warns if config.py differs). The final chain is then run with config.py
on the training AND the test patients; test numbers are reported separately and
never used to choose anything.

Metrics (src/metrics.py):
    CNR  = |mean(kidney) - mean(ring around kidney)| / std(ring)       higher = better
    PSNR = closeness to the windowed input (NOT a noise-removal score)  higher = less changed
    edge ratio = kidney-boundary gradient after / before                 < 1 = edges blurred

Outputs:
    results/tables/step2_window_comparison.csv
    results/tables/step2_denoising_comparison.csv
    results/tables/step2_frequency_comparison.csv
    results/tables/step2_contrast_comparison.csv
    results/tables/step2_final_chain_test_results.csv
    results/figures/step2_window_comparison.png
    results/figures/step2_denoising_comparison.png      (with zoom on the kidney edge)
    results/figures/step2_diffusion_sweep.png
    results/figures/step2_he_vs_clahe.png               (with histograms)
    results/figures/step2_homomorphic_spectrum.png      (spectra before/after + filter shape)
    results/figures/step2_full_chain_train_example.png
    results/figures/step2_full_chain_case22_test.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to file without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

import config  # noqa: E402
from src import data_loading, metrics  # noqa: E402
from src import phase1_enhancement as enh  # noqa: E402

METRIC_COLUMNS = ["cnr", "psnr", "edge_ratio"]


# =============================================================================
# Helpers
# =============================================================================
def load_slices_with_rings(split: str) -> list[dict]:
    """Load one split and add the CNR background ring to every slice."""
    slices = data_loading.load_split(split)
    for slice_data in slices:
        slice_data["ring"] = metrics.make_background_ring(slice_data["ground_truth"], config.CNR_BACKGROUND_RING_PX)
    return slices


def score_image(image: np.ndarray, reference: np.ndarray, slice_data: dict) -> dict:
    """CNR of the image, plus PSNR and edge-strength ratio relative to the reference image."""
    kidney = slice_data["ground_truth"]
    return {
        "cnr": metrics.cnr(image, kidney, slice_data["ring"]),
        "psnr": metrics.psnr(reference, image),
        "edge_ratio": metrics.edge_strength_on_boundary(image, kidney)
        / metrics.edge_strength_on_boundary(reference, kidney),
    }


def summarise(per_slice: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    """Mean, standard deviation and median of each metric over slices, one row per setting."""
    # Median is reported too, and used to pick winners (config.SELECTION_STATISTIC): a few
    # slices whose background ring is almost all clipped fat get huge CNRs that dominate the mean.
    summary = per_slice.groupby(group_columns, sort=False, dropna=False)[metric_columns].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.insert(0, "n_slices", per_slice.groupby(group_columns, sort=False, dropna=False).size())
    return summary.reset_index()


def save_table(table: pd.DataFrame, name: str) -> None:
    """Write a CSV to results/tables/ with numbers rounded for readability."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(config.TABLES_DIR / name, index=False)
    print(f"    saved results/tables/{name}")


def save_figure(figure, name: str) -> None:
    """Write a PNG to results/figures/ and close it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, 0.95))  # leave room for the suptitle; stop labels overlapping
    figure.savefig(config.FIGURES_DIR / name, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"    saved results/figures/{name}")


def find_slice(slices: list[dict], case_id: int, slice_index: int) -> dict:
    """Pick one slice dict by case and slice number."""
    for slice_data in slices:
        if slice_data["case_id"] == case_id and slice_data["slice_index"] == slice_index:
            return slice_data
    raise KeyError(f"case {case_id} slice {slice_index} not found")


def kidney_zoom_box(kidney_mask: np.ndarray, margin: int = 20) -> tuple[int, int, int, int]:
    """Square box (row0, row1, col0, col1) around the largest kidney piece, for edge close-ups."""
    labelled, _ = ndimage.label(kidney_mask)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    rows, cols = np.nonzero(labelled == sizes.argmax())
    centre_row, centre_col = (rows.min() + rows.max()) // 2, (cols.min() + cols.max()) // 2
    half = max(rows.max() - rows.min(), cols.max() - cols.min()) // 2 + margin
    return (max(centre_row - half, 0), centre_row + half, max(centre_col - half, 0), centre_col + half)


def show(axis, image, title, box=None, **imshow_options) -> None:
    """Display an image without axes, optionally cropped to a zoom box."""
    if box is not None:
        row0, row1, col0, col1 = box
        image = image[row0:row1, col0:col1]
    imshow_options.setdefault("cmap", "gray")
    axis.imshow(image, **imshow_options)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def best_row(table: pd.DataFrame, metric: str = "cnr") -> pd.Series:
    """Row with the highest training score, using config.SELECTION_STATISTIC (median)."""
    return table.loc[table[f"{metric}_{config.SELECTION_STATISTIC}"].idxmax()]


def window_bounds_to_level_width(lowest_hu: float, highest_hu: float) -> tuple[float, float]:
    """Convert a window written as [lowest, highest] HU into (level, width)."""
    return (lowest_hu + highest_hu) / 2, highest_hu - lowest_hu


# =============================================================================
# 1. HU window
# =============================================================================
def compare_windows(train_slices: list[dict]) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Score each candidate window on the training slices; return the table and the best window.

    Besides CNR we record how much of the kidney / ring is clipped to pure black or
    white, because CNR can only change between windows through clipping (a linear
    stretch alone leaves CNR unchanged).
    """
    rows = []
    for lowest_hu, highest_hu in config.HU_WINDOW_CANDIDATES:
        level, width = window_bounds_to_level_width(lowest_hu, highest_hu)
        for slice_data in train_slices:
            windowed = enh.apply_hu_window(slice_data["hu"], level, width)
            kidney_values = windowed[slice_data["ground_truth"]]
            ring_values = windowed[slice_data["ring"]]
            rows.append({
                "window": f"[{lowest_hu}, {highest_hu}]", "level": level, "width": width,
                "cnr": metrics.cnr(windowed, slice_data["ground_truth"], slice_data["ring"]),
                "kidney_clipped_white_pct": 100 * np.mean(kidney_values == 255),
                "kidney_clipped_black_pct": 100 * np.mean(kidney_values == 0),
                "ring_clipped_black_pct": 100 * np.mean(ring_values == 0),
            })
    per_slice = pd.DataFrame(rows)
    table = summarise(per_slice, ["window", "level", "width"],
                      ["cnr", "kidney_clipped_white_pct", "kidney_clipped_black_pct", "ring_clipped_black_pct"])
    best = best_row(table)
    return table, (float(best["level"]), float(best["width"]))


# =============================================================================
# 2. Noise removal
# =============================================================================
def compare_denoising(train_slices: list[dict], windowed: list[np.ndarray]) -> pd.DataFrame:
    """Run every median kernel and every diffusion setting on every training slice."""
    rows = []
    for slice_data, image in zip(train_slices, windowed):
        for kernel in config.MEDIAN_KERNEL_CANDIDATES:
            result = enh.median_filter(image, kernel)
            rows.append({"method": "median", "setting": f"kernel {kernel}", "kernel": kernel,
                         **score_image(result, image, slice_data)})
        for conduction in config.ANISO_CONDUCTION_CANDIDATES:
            for kappa in config.ANISO_KAPPA_CANDIDATES:
                for iterations in config.ANISO_ITERATION_CANDIDATES:
                    result = enh.anisotropic_diffusion_perona_malik(image, iterations, kappa,
                                                                     config.ANISO_GAMMA, conduction)
                    rows.append({"method": "anisotropic diffusion",
                                 "setting": f"kappa {kappa}, {iterations} it, g{conduction}",
                                 "kappa": kappa, "iterations": iterations, "conduction": conduction,
                                 **score_image(result, image, slice_data)})
    per_slice = pd.DataFrame(rows)
    return summarise(per_slice, ["method", "setting", "kernel", "kappa", "iterations", "conduction"],
                     METRIC_COLUMNS)


def pick_best_smoothing(table: pd.DataFrame, method: str) -> pd.Series:
    """Highest mean CNR among one method's settings that keep enough edge strength."""
    rows = table[table["method"] == method]
    acceptable = rows[rows[f"edge_ratio_{config.SELECTION_STATISTIC}"] >= config.MIN_EDGE_STRENGTH_RATIO]
    if acceptable.empty:
        print(f"    WARNING: no {method} setting keeps {config.MIN_EDGE_STRENGTH_RATIO:.0%} edge strength")
        acceptable = rows
    return best_row(acceptable)


# =============================================================================
# 3. Homomorphic filter
# =============================================================================
def compare_homomorphic(train_slices: list[dict], denoised: list[np.ndarray]) -> pd.DataFrame:
    """Run every homomorphic setting on the denoised training slices.

    gamma_low = gamma_high = 1 makes H = 1 everywhere (the filter does nothing); that
    row is kept once as the "no filter" reference. The winner is the best REAL filter
    by CNR. The edge ratio is not used here: the output is re-stretched to 0-255,
    which changes every gradient's size, so the ratio mixes sharpness with scaling.
    """
    settings = [(1.0, 1.0, None)]  # the do-nothing reference
    for gamma_low in config.HOMOMORPHIC_GAMMA_LOW_CANDIDATES:
        for gamma_high in config.HOMOMORPHIC_GAMMA_HIGH_CANDIDATES:
            if gamma_low == gamma_high == 1.0:
                continue
            for cutoff in config.HOMOMORPHIC_CUTOFF_D0_CANDIDATES:
                settings.append((gamma_low, gamma_high, cutoff))

    rows = []
    for slice_data, image in zip(train_slices, denoised):
        for gamma_low, gamma_high, cutoff in settings:
            result = enh.homomorphic_filter(image, gamma_low, gamma_high, cutoff or 30, config.HOMOMORPHIC_SHARPNESS_C)
            label = "no filter (gL = gH = 1)" if cutoff is None else f"gL {gamma_low}, gH {gamma_high}, D0 {cutoff}"
            rows.append({"setting": label, "gamma_low": gamma_low, "gamma_high": gamma_high,
                         "cutoff_d0": cutoff, **score_image(result, image, slice_data)})
    per_slice = pd.DataFrame(rows)
    return summarise(per_slice, ["setting", "gamma_low", "gamma_high", "cutoff_d0"], METRIC_COLUMNS)


# =============================================================================
# 4. Contrast: HE vs CLAHE
# =============================================================================
def compare_contrast(train_slices: list[dict], inputs: list[np.ndarray]) -> pd.DataFrame:
    """Global histogram equalisation vs every CLAHE setting, plus the unchanged input."""
    rows = []
    for slice_data, image in zip(train_slices, inputs):
        rows.append({"method": "none (input)", "setting": "-", **score_image(image, image, slice_data)})
        rows.append({"method": "histogram equalisation", "setting": "global",
                     **score_image(enh.histogram_equalisation(image), image, slice_data)})
        for clip_limit in config.CLAHE_CLIP_CANDIDATES:
            for tiles in config.CLAHE_TILE_CANDIDATES:
                result = enh.clahe_enhancement(image, clip_limit, tiles)
                rows.append({"method": "CLAHE", "setting": f"clip {clip_limit}, tiles {tiles[0]}x{tiles[1]}",
                             "clip_limit": clip_limit, "tiles": tiles[0],
                             **score_image(result, image, slice_data)})
    per_slice = pd.DataFrame(rows)
    return summarise(per_slice, ["method", "setting", "clip_limit", "tiles"], METRIC_COLUMNS)


# =============================================================================
# 5. Final chain (uses config.py values) on train and test
# =============================================================================
def run_chain_stages(hu_slice: np.ndarray) -> dict:
    """Every intermediate image of the final chain, using the settings in config.py."""
    windowed = enh.apply_hu_window(hu_slice)
    denoised = enh.anisotropic_diffusion_perona_malik(windowed)
    homomorphic = enh.homomorphic_filter(denoised)
    final = enh.clahe_enhancement(homomorphic)
    return {"1 window": windowed, "2 + diffusion": denoised, "3 + homomorphic": homomorphic, "4 + CLAHE (final)": final}


def evaluate_final_chain(slices_by_split: dict) -> pd.DataFrame:
    """CNR / PSNR / edge ratio after each stage, against the windowed slice, per split and per patient."""
    rows = []
    for split, slices in slices_by_split.items():
        for slice_data in slices:
            stages = run_chain_stages(slice_data["hu"])
            windowed = stages["1 window"]
            # Raw HU CNR: the starting point before any processing.
            rows.append({"split": split, "case_id": slice_data["case_id"], "stage": "0 raw HU",
                         "cnr": metrics.cnr(slice_data["hu"], slice_data["ground_truth"], slice_data["ring"]),
                         "psnr": np.nan, "edge_ratio": np.nan})
            for stage_name, image in stages.items():
                rows.append({"split": split, "case_id": slice_data["case_id"], "stage": stage_name,
                             **score_image(image, windowed, slice_data)})
    per_slice = pd.DataFrame(rows)

    overall = summarise(per_slice, ["split", "stage"], METRIC_COLUMNS)
    overall.insert(1, "case_id", "all")
    per_patient = summarise(per_slice[per_slice["split"] == "test"], ["split", "case_id", "stage"], METRIC_COLUMNS)
    return pd.concat([overall, per_patient], ignore_index=True)


# =============================================================================
# Figures
# =============================================================================
def figure_windows(slice_data: dict) -> None:
    """The three candidate windows side by side on one training slice."""
    figure, axes = plt.subplots(1, len(config.HU_WINDOW_CANDIDATES), figsize=(4 * len(config.HU_WINDOW_CANDIDATES), 4.3))
    for axis, (lowest_hu, highest_hu) in zip(axes, config.HU_WINDOW_CANDIDATES):
        windowed = enh.apply_hu_window(slice_data["hu"], *window_bounds_to_level_width(lowest_hu, highest_hu))
        value = metrics.cnr(windowed, slice_data["ground_truth"], slice_data["ring"])
        show(axis, windowed, f"window [{lowest_hu}, {highest_hu}] HU\nCNR {value:.2f}")
    figure.suptitle(f"HU windows - case {slice_data['case_id']} slice {slice_data['slice_index']} (training)")
    save_figure(figure, "step2_window_comparison.png")


def figure_denoising(slice_data: dict) -> None:
    """Median 3/5/7 vs the chosen diffusion: full slice, kidney-edge zoom, and what each filter removed."""
    windowed = enh.apply_hu_window(slice_data["hu"])
    results = {"windowed input": windowed}
    for kernel in config.MEDIAN_KERNEL_CANDIDATES:
        results[f"median {kernel}x{kernel}"] = enh.median_filter(windowed, kernel)
    results[f"anisotropic diffusion\nkappa {config.ANISO_KAPPA}, {config.ANISO_ITERATIONS} it, g{config.ANISO_CONDUCTION}"] = \
        enh.anisotropic_diffusion_perona_malik(windowed)

    box = kidney_zoom_box(slice_data["ground_truth"])
    figure, axes = plt.subplots(3, len(results), figsize=(3.2 * len(results), 10))
    for column, (name, image) in enumerate(results.items()):
        scores = score_image(image, windowed, slice_data)
        psnr_text = "inf" if np.isinf(scores["psnr"]) else f"{scores['psnr']:.1f} dB"
        show(axes[0, column], image, f"{name}\nCNR {scores['cnr']:.2f} | PSNR {psnr_text} | edge {scores['edge_ratio']:.2f}")
        axes[0, column].add_patch(Rectangle((box[2], box[0]), box[3] - box[2], box[1] - box[0],
                                            fill=False, edgecolor="yellow", linewidth=0.8))
        show(axes[1, column], image, "zoom on kidney edge", box)
        # What the filter took away: ideally only grainy noise, no kidney outline.
        removed = windowed.astype(float) - image.astype(float)
        show(axes[2, column], removed, "removed (input - output)", box, cmap="seismic", vmin=-40, vmax=40)
    figure.suptitle(f"Noise removal - case {slice_data['case_id']} slice {slice_data['slice_index']} (training). "
                    "Bottom row: an outline of the kidney = the filter removed edge, not just noise.", fontsize=10)
    save_figure(figure, "step2_denoising_comparison.png")


def figure_diffusion_sweep(denoise_table: pd.DataFrame) -> None:
    """Heatmaps of mean training CNR and edge ratio over kappa x iterations (chosen conduction)."""
    rows = denoise_table[(denoise_table["method"] == "anisotropic diffusion")
                         & (denoise_table["conduction"] == config.ANISO_CONDUCTION)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, column, title in [(axes[0], "cnr_median", "median CNR (higher = better)"),
                                (axes[1], "edge_ratio_median", "median edge-strength ratio (1 = edges kept)")]:
        grid = rows.pivot(index="kappa", columns="iterations", values=column)
        image = axis.imshow(grid.values, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(grid.columns)), [int(v) for v in grid.columns])
        axis.set_yticks(range(len(grid.index)), [int(v) for v in grid.index])
        axis.set_xlabel("iterations")
        axis.set_ylabel("kappa")
        axis.set_title(title, fontsize=9)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                axis.text(c, r, f"{grid.values[r, c]:.2f}", ha="center", va="center", color="white", fontsize=8)
        figure.colorbar(image, ax=axis)
    figure.suptitle(f"Perona-Malik sweep on 48 training slices (conduction g{config.ANISO_CONDUCTION}, "
                    f"step {config.ANISO_GAMMA}). Chosen: kappa {config.ANISO_KAPPA}, {config.ANISO_ITERATIONS} iterations")
    save_figure(figure, "step2_diffusion_sweep.png")


def figure_he_vs_clahe(slice_data: dict) -> None:
    """Input vs global HE vs chosen CLAHE, with histograms of the pixels inside the body."""
    stages = run_chain_stages(slice_data["hu"])
    contrast_input = stages["3 + homomorphic"]
    images = {"input (after diffusion + homomorphic)": contrast_input,
              "global histogram equalisation": enh.histogram_equalisation(contrast_input),
              f"CLAHE clip {config.CLAHE_CLIP_LIMIT}, tiles {config.CLAHE_TILE_GRID[0]}x{config.CLAHE_TILE_GRID[1]}":
                  enh.clahe_enhancement(contrast_input)}
    # Histogram of body pixels only: the black air outside the patient would dwarf everything else.
    body = slice_data["hu"] > config.BODY_HU_THRESHOLD

    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for column, (name, image) in enumerate(images.items()):
        value = metrics.cnr(image, slice_data["ground_truth"], slice_data["ring"])
        show(axes[0, column], image, f"{name}\nCNR {value:.2f}")
        axes[1, column].hist(image[body], bins=64, range=(0, 255), color="grey")
        axes[1, column].hist(image[slice_data["ground_truth"]], bins=64, range=(0, 255), color="green", alpha=0.7)
        axes[1, column].set_title("histogram inside the body (green = kidney)", fontsize=8)
        axes[1, column].set_xlabel("grey level")
    figure.suptitle(f"Histogram equalisation vs CLAHE - case {slice_data['case_id']} slice {slice_data['slice_index']} (training)")
    save_figure(figure, "step2_he_vs_clahe.png")


def figure_homomorphic(slice_data: dict) -> None:
    """Log spectrum before/after the homomorphic filter, the filter H itself, and the images."""
    stages = run_chain_stages(slice_data["hu"])
    before, after = stages["2 + diffusion"], stages["3 + homomorphic"]
    high_emphasis = enh.gaussian_high_emphasis_filter(before.shape)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    show(axes[0, 0], before, "input (after diffusion)")
    show(axes[0, 1], enh.dft_log_magnitude_spectrum(before), "log-magnitude spectrum BEFORE", cmap="magma")
    show(axes[0, 2], enh.dft_log_magnitude_spectrum(after), "log-magnitude spectrum AFTER", cmap="magma")
    show(axes[1, 0], after, "output of homomorphic filter")
    show(axes[1, 1], high_emphasis, f"filter H(u,v): gL {config.HOMOMORPHIC_GAMMA_LOW}, "
                                    f"gH {config.HOMOMORPHIC_GAMMA_HIGH}, D0 {config.HOMOMORPHIC_CUTOFF_D0}", cmap="viridis")
    centre_row = high_emphasis.shape[0] // 2
    profile = high_emphasis[centre_row, high_emphasis.shape[1] // 2:]
    axes[1, 2].plot(profile)
    axes[1, 2].axvline(config.HOMOMORPHIC_CUTOFF_D0, linestyle="--", color="grey", label="D0")
    axes[1, 2].set_xlabel("distance D from centre (frequency pixels)")
    axes[1, 2].set_ylabel("gain H")
    axes[1, 2].set_title("filter profile: low frequencies x gL, high x gH", fontsize=8)
    axes[1, 2].legend()
    figure.suptitle(f"Homomorphic filter - case {slice_data['case_id']} slice {slice_data['slice_index']} (training)")
    save_figure(figure, "step2_homomorphic_spectrum.png")


def figure_full_chain(slice_data: dict, file_name: str, split_label: str) -> None:
    """Raw HU -> window -> diffusion -> homomorphic -> CLAHE, full slice and kidney zoom."""
    stages = {"0 raw HU (full range)": slice_data["hu"], **run_chain_stages(slice_data["hu"])}
    box = kidney_zoom_box(slice_data["ground_truth"])
    figure, axes = plt.subplots(2, len(stages), figsize=(3.3 * len(stages), 7))
    for column, (name, image) in enumerate(stages.items()):
        value = metrics.cnr(image, slice_data["ground_truth"], slice_data["ring"])
        show(axes[0, column], image, f"{name}\nCNR {value:.2f}")
        show(axes[1, column], image, "zoom on kidney", box)
        axes[1, column].contour(slice_data["ground_truth"][box[0]:box[1], box[2]:box[3]], levels=[0.5],
                                colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=0.6)
    figure.suptitle(f"Full enhancement chain - case {slice_data['case_id']} slice {slice_data['slice_index']} "
                    f"({split_label}). Green = ground-truth kidney.")
    save_figure(figure, file_name)


# =============================================================================
# Main
# =============================================================================
def report_choice(name: str, chosen_on_training, value_in_config) -> None:
    """Print the training winner next to what config.py holds, flagging any mismatch."""
    def plain(value):  # np.float64(10.0) -> 10.0 so the printout is readable
        return tuple(float(v) for v in value) if isinstance(value, tuple) else float(value)
    chosen_on_training, value_in_config = plain(chosen_on_training), plain(value_in_config)
    status = "OK" if chosen_on_training == value_in_config else "<-- UPDATE config.py"
    print(f"    {name:32s} training winner: {str(chosen_on_training):18s} config.py: {str(value_in_config):18s} {status}")


def main() -> None:
    train_slices = load_slices_with_rings("train")
    test_slices = load_slices_with_rings("test")
    print(f"Training slices: {len(train_slices)}   Test slices: {len(test_slices)} (test used only at the end)\n")

    # ---- 1. Window ----
    print("1) HU window")
    window_table, (best_level, best_width) = compare_windows(train_slices)
    save_table(window_table, "step2_window_comparison.csv")
    print(window_table[["window", "cnr_mean", "cnr_std", "cnr_median", "kidney_clipped_white_pct_mean",
                        "ring_clipped_black_pct_mean"]]
          .round(2).to_string(index=False))
    windowed = [enh.apply_hu_window(s["hu"], best_level, best_width) for s in train_slices]

    # ---- 2. Noise removal ----
    print("\n2) Noise removal (this sweep takes a few minutes)")
    denoise_table = compare_denoising(train_slices, windowed)
    save_table(denoise_table, "step2_denoising_comparison.csv")
    best_diffusion = pick_best_smoothing(denoise_table, "anisotropic diffusion")
    best_median = pick_best_smoothing(denoise_table, "median")
    for label, row in [("best diffusion", best_diffusion), ("best median", best_median)]:
        print(f"    {label:15s} {row['setting']:26s} CNR {row['cnr_mean']:.2f} +/- {row['cnr_std']:.2f} (median {row['cnr_median']:.2f}) | "
              f"PSNR {row['psnr_median']:.1f} dB | edge ratio {row['edge_ratio_median']:.3f}")
    denoised = [enh.anisotropic_diffusion_perona_malik(image, int(best_diffusion["iterations"]),
                                                       best_diffusion["kappa"], config.ANISO_GAMMA,
                                                       int(best_diffusion["conduction"])) for image in windowed]

    # ---- 3. Homomorphic ----
    print("\n3) Homomorphic filter")
    frequency_table = compare_homomorphic(train_slices, denoised)
    save_table(frequency_table, "step2_frequency_comparison.csv")
    no_filter = frequency_table[frequency_table["cutoff_d0"].isna()].iloc[0]
    real_filters = frequency_table[frequency_table["cutoff_d0"].notna()]
    best_homomorphic = best_row(real_filters)
    for label, row in [("no filter", no_filter), ("best filter", best_homomorphic)]:
        print(f"    {label:12s} {row['setting']:28s} CNR {row['cnr_mean']:.2f} +/- {row['cnr_std']:.2f} (median {row['cnr_median']:.2f}) | "
              f"PSNR {row['psnr_median']:.1f} dB | edge ratio {row['edge_ratio_median']:.3f}")
    homomorphic_outputs = [enh.homomorphic_filter(image, best_homomorphic["gamma_low"], best_homomorphic["gamma_high"],
                                                  best_homomorphic["cutoff_d0"], config.HOMOMORPHIC_SHARPNESS_C)
                           for image in denoised]

    # ---- 4. Contrast ----
    print("\n4) Contrast: histogram equalisation vs CLAHE")
    contrast_table = compare_contrast(train_slices, homomorphic_outputs)
    save_table(contrast_table, "step2_contrast_comparison.csv")
    clahe_rows = contrast_table[contrast_table["method"] == "CLAHE"]
    best_clahe = best_row(clahe_rows)
    for label, row in [("input", contrast_table.iloc[0]), ("global HE", contrast_table.iloc[1]), ("best CLAHE", best_clahe)]:
        print(f"    {label:10s} {row['setting']:26s} CNR {row['cnr_mean']:.2f} +/- {row['cnr_std']:.2f} (median {row['cnr_median']:.2f}) | "
              f"edge ratio {row['edge_ratio_median']:.3f}")

    # ---- Training winners vs config.py ----
    print("\nChosen on training vs config.py:")
    report_choice("HU window (level, width)", (best_level, best_width),
                  (float(config.HU_WINDOW_LEVEL), float(config.HU_WINDOW_WIDTH)))
    report_choice("diffusion (kappa, iter, g)",
                  (best_diffusion["kappa"], int(best_diffusion["iterations"]), int(best_diffusion["conduction"])),
                  (config.ANISO_KAPPA, config.ANISO_ITERATIONS, config.ANISO_CONDUCTION))
    report_choice("median kernel", int(best_median["kernel"]), config.MEDIAN_KERNEL_SIZE)
    report_choice("homomorphic (gL, gH, D0)",
                  (best_homomorphic["gamma_low"], best_homomorphic["gamma_high"], best_homomorphic["cutoff_d0"]),
                  (config.HOMOMORPHIC_GAMMA_LOW, config.HOMOMORPHIC_GAMMA_HIGH, float(config.HOMOMORPHIC_CUTOFF_D0)))
    report_choice("CLAHE (clip, tiles)", (best_clahe["clip_limit"], int(best_clahe["tiles"])),
                  (config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID[0]))

    # ---- Final chain with config.py values: train, then test ----
    print("\n5) Final chain (config.py settings)")
    chain_table = evaluate_final_chain({"train": train_slices, "test": test_slices})
    save_table(chain_table, "step2_final_chain_test_results.csv")
    for split in ["train", "test"]:
        print(f"    {split.upper()} (all patients):")
        rows = chain_table[(chain_table["split"] == split) & (chain_table["case_id"] == "all")]
        print(rows[["stage", "cnr_mean", "cnr_std", "cnr_median", "psnr_median", "edge_ratio_median"]]
              .round(2).to_string(index=False))
    print("    TEST per patient (final stage only):")
    final_rows = chain_table[(chain_table["case_id"] != "all") & (chain_table["stage"] == "4 + CLAHE (final)")]
    print(final_rows[["case_id", "cnr_mean", "cnr_std", "cnr_median", "edge_ratio_median"]].round(2).to_string(index=False))

    # ---- Figures ----
    print("\n6) Figures")
    example = find_slice(train_slices, *config.EXAMPLE_TRAIN_SLICE)
    hard_test = find_slice(test_slices, *config.EXAMPLE_HARD_TEST_SLICE)
    figure_windows(example)
    figure_denoising(example)
    figure_diffusion_sweep(denoise_table)
    figure_he_vs_clahe(example)
    figure_homomorphic(example)
    figure_full_chain(example, "step2_full_chain_train_example.png", "training")
    figure_full_chain(hard_test, "step2_full_chain_case22_test.png", "TEST - hardest patient, 0.5 mm slices")


if __name__ == "__main__":
    main()
