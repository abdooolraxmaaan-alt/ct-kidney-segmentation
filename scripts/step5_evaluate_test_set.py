"""
Step 5 - final evaluation of the finished pipeline. Uses config.py and models/ only:
NO setting is changed or chosen here.

Run from the project folder:  .venv/bin/python scripts/step5_evaluate_test_set.py
(needs models/ from steps 3 and 4, and results/tables/step4_candidate_predictions.csv)

What it does:
    1. runs the final pipeline end to end on the 24 TEST slices
       (raw HU -> Phase 1 -> region growing + rule filter -> k-NN check with P(kidney)),
       timing each slice
    2. one consolidated table of every phase's metrics, train vs test
       (Phase 1 PSNR/CNR, Phase 2 Dice/IoU for both strategies, Phase 3 classifier scores;
       training classifier scores are nested out-of-fold)
    3. per-patient and per-slice tables (predicted vs ground-truth areas)
    4. figures: pipeline block diagram, best/median/worst test slices, Dice per patient
    5. results/failure_analysis.md: every test slice with Dice < 0.85 grouped by cause,
       with an example figure and a realistic fix for each cause
    6. exports the 24 test slices to sample_test_images/ (.npz + PNG preview + README)

Outputs:
    results/tables/final_results_summary.csv
    results/tables/final_results_per_patient.csv
    results/tables/final_results_per_slice.csv
    results/figures/final_pipeline_block_diagram.png
    results/figures/final_test_results_panel.png
    results/figures/final_dice_per_patient.png
    results/figures/failure_<cause>.png
    results/failure_analysis.md
    sample_test_images/
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to file without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

import config  # noqa: E402
from src import data_loading, metrics  # noqa: E402
from src import phase1_enhancement as enh  # noqa: E402
from src import phase2_segmentation as seg  # noqa: E402
from src.pipeline import plot_contour_overlay, segment_kidney_with_details  # noqa: E402

FINAL = config.FINAL_SEGMENTATION_STRATEGY
SIDE_NAMES = {"patient_right": "right", "patient_left": "left"}

# Failure causes, in the order they are checked, with an explanation and a realistic fix.
FAILURE_CAUSES = {
    "sliver kidney": (
        "The ground-truth kidney on one side is only a thin tip (top or bottom pole, under "
        f"{config.FAILURE_SLIVER_AREA_PX} pixels). It is either missed or a few pixels of boundary error "
        "dominate the Dice, because Dice is very sensitive on tiny objects.",
        "Use 3D context: segment the neighbouring slices first and carry the kidney position into the pole "
        "slices; report pole slices separately, or use a size-aware metric (e.g. boundary distance) for them."),
    "seed outside the kidney": (
        "The learned seed position (spine centroid + average offset) fell outside this kidney, so the region "
        "grew in another organ or nothing kidney-like was found nearby. Kidney position varies with slice level "
        "and between patients, but one average offset is used for every slice.",
        "Learn seed offsets as a function of slice level (upper pole / hilum / lower pole), or place several "
        "candidate seeds per side and let the k-NN pick the region that looks most like kidney."),
    "tumour not captured": (
        "Most of the missed kidney is tumour (KiTS label 2). The tumour's density differs from the enhancing "
        "cortex the seed sits on, so region growing stops at the tumour edge.",
        "Grow from a second seed inside the darker part enclosed by the kidney, or merge this mask with the "
        "output of a dedicated tumour segmentation so the whole-kidney outline includes the tumour."),
    "leak": (
        "The region grew beyond the kidney, usually along the renal vessels toward the spine or into touching "
        "bowel of similar brightness after equalisation. Either the leaked region is kept (precision below 75%), "
        "or it grew so large that the rule filter rejected it as not kidney-shaped and that side ends up empty "
        "(yellow = grown region in the figure). Kidneys with a large tumour are most at risk: their mixed "
        "density lets the running mean drift.",
        "Stop growth at strong edges (a gradient barrier) as well as on intensity, use a smaller tolerance once "
        "the region reaches the hilum, or open with a larger kidney-shaped structuring element."),
    "boundary disagreement": (
        "Both kidneys were found, but the outline differs from the expert's: the ground truth leaves the renal "
        "sinus fat out at the hilum while our hole filling includes it, and the blurred partial-volume edge is "
        "split differently.",
        "Do not fill the hilum notch (fill only fully enclosed holes away from the medial edge), and refine the "
        "final boundary on the un-equalised image."),
}


# =============================================================================
# Helpers
# =============================================================================
def save_table(table: pd.DataFrame, name: str) -> None:
    """Write a CSV to results/tables/ with numbers rounded for readability."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(config.TABLES_DIR / name, index=False)
    print(f"    saved results/tables/{name}")


def save_figure(figure, name: str) -> None:
    """Write a PNG to results/figures/ and close it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.FIGURES_DIR / name, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"    saved results/figures/{name}")


def statistics(values, unit: str) -> dict:
    """Mean, median and standard deviation of a list of numbers."""
    values = np.asarray(values, dtype=float)
    return {"mean": values.mean(), "median": np.median(values), "std": values.std(ddof=1) if len(values) > 1 else 0.0,
            "n": len(values), "unit": unit}


def strategy_settings(strategy: str) -> dict:
    """The settings config.py holds for one strategy."""
    settings = {"opening_radius": config.OPENING_RADIUS_FOR_STRATEGY[strategy]}
    if strategy == "multi_otsu":
        settings.update(n_classes=config.MULTI_OTSU_CLASSES, n_kidney_classes=config.MULTI_OTSU_KIDNEY_CLASSES)
    else:
        settings.update(tolerance=config.REGION_GROW_TOLERANCE)
    return settings


# =============================================================================
# 1. Running the pipeline
# =============================================================================
def evaluate_slice(slice_data: dict) -> dict:
    """Run the final pipeline (timed) plus the measurements every table needs, for one slice."""
    hu, truth = slice_data["hu"], slice_data["ground_truth"]

    start = time.perf_counter()
    details = segment_kidney_with_details(hu)          # the exact function other programs call
    runtime = time.perf_counter() - start

    # The same steps again, to keep intermediate results (seeds, stages) for analysis and figures.
    anatomy = seg.find_anatomy(hu)
    windowed = enh.apply_hu_window(hu)
    denoised = enh.anisotropic_diffusion_perona_malik(windowed)
    homomorphic = enh.homomorphic_filter(denoised)
    inputs = {s: enh.apply_contrast_step(homomorphic, config.CONTRAST_FOR_STRATEGY[s])
              for s in ("multi_otsu", "region_growing")}
    results = {s: seg.segment_slice(inputs[s], anatomy, details_priors(), s, config.CONTRAST_FOR_STRATEGY[s],
                                    **strategy_settings(s)) for s in ("multi_otsu", "region_growing")}
    if not np.array_equal(results[FINAL]["final"], details["mask"]):
        raise RuntimeError("re-run of the final strategy does not match segment_kidney - check config")

    ring = metrics.make_background_ring(truth, config.CNR_BACKGROUND_RING_PX)
    stages = {"0 raw HU": hu, "1 window": windowed, "2 + diffusion": denoised, "3 + homomorphic": homomorphic,
              f"4 + {config.SEGMENTATION_CONTRAST} (final input)": inputs[FINAL]}
    phase1 = {name: {"cnr": metrics.cnr(image, truth, ring),
                     # PSNR is measured against the windowed slice, so it is undefined for raw HU and
                     # infinite for the windowed slice itself: left empty for both.
                     "psnr": np.nan if name.startswith(("0", "1")) else metrics.psnr(windowed, image)}
              for name, image in stages.items()}
    return {"slice": slice_data, "details": details, "runtime": runtime, "anatomy": anatomy,
            "windowed": windowed, "final_input": inputs[FINAL], "results": results, "phase1": phase1}


_priors = {}


def details_priors() -> dict:
    """The learned Phase 2 priors (loaded once)."""
    if not _priors:
        _priors.update(seg.load_priors())
    return _priors


# =============================================================================
# 2. Failure causes
# =============================================================================
def failure_cause(evaluation: dict) -> str:
    """One primary cause for a slice's errors, checked in the order of FAILURE_CAUSES."""
    slice_data, result = evaluation["slice"], evaluation["results"][FINAL]
    predicted, truth = evaluation["details"]["mask"], slice_data["ground_truth"]
    anatomy = evaluation["anatomy"]
    truth_by_side = seg.ground_truth_kidney_per_side(truth, anatomy)
    seeds = {s["side"]: s for s in result["seeds"]}

    # Which side loses the most kidney pixels?
    column_index = np.indices(truth.shape)[1]
    spine_col = anatomy["spine"][1]
    worst_side, worst_missed = None, 0
    for side, zone in zip(seg.SIDES, [column_index < spine_col, column_index >= spine_col]):
        missed = (truth & zone & ~predicted).sum() + (predicted & zone & ~truth).sum()
        if missed > worst_missed:
            worst_side, worst_missed = side, missed

    piece = truth_by_side.get(worst_side)
    if piece is not None and piece.sum() < config.FAILURE_SLIVER_AREA_PX:
        return "sliver kidney"
    if piece is not None and (predicted & piece).sum() < 0.1 * piece.sum():
        seed = seeds.get(worst_side, {}).get("seed")
        if seed is None or not piece[seed]:
            return "seed outside the kidney"
    overlap = (predicted & truth).sum()
    precision = overlap / predicted.sum() if predicted.any() else 0.0
    recall = overlap / truth.sum()
    missed_pixels = truth & ~predicted
    if recall < 0.75 and (slice_data["label_map"][missed_pixels] == config.TUMOUR_LABEL).mean() > 0.5:
        return "tumour not captured"
    if precision < 0.75 or (piece is not None and (predicted & piece).sum() < 0.1 * piece.sum()):
        return "leak"
    return "boundary disagreement"


# =============================================================================
# 3. Tables
# =============================================================================
def per_slice_table(evaluations: list[dict]) -> pd.DataFrame:
    """One row per slice: Dice, IoU, areas, P(kidney) of each final region, runtime and cause."""
    rows = []
    for evaluation in evaluations:
        slice_data, details = evaluation["slice"], evaluation["details"]
        predicted, truth = details["mask"], slice_data["ground_truth"]
        pixel_area_mm2 = slice_data["pixel_spacing"][0] * slice_data["pixel_spacing"][1]
        dice = metrics.dice_coefficient(predicted, truth)
        rows.append({
            "case_id": slice_data["case_id"], "slice_index": slice_data["slice_index"], "split": slice_data["split"],
            "dice": dice, "iou": metrics.iou_score(predicted, truth),
            "dice_multi_otsu": metrics.dice_coefficient(evaluation["results"]["multi_otsu"]["final"], truth),
            "predicted_area_px": int(predicted.sum()), "truth_area_px": int(truth.sum()),
            "predicted_area_mm2": round(predicted.sum() * pixel_area_mm2, 1),
            "truth_area_mm2": round(truth.sum() * pixel_area_mm2, 1),
            "area_error_pct": 100 * (predicted.sum() - truth.sum()) / truth.sum(),
            "n_regions": len(details["regions"]),
            "regions_p_kidney": "; ".join(f"{r['side']} {r['p_kidney']:.2f}" for r in details["regions"]),
            "tumour_in_slice": bool((slice_data["label_map"] == config.TUMOUR_LABEL).any()),
            "runtime_s": evaluation["runtime"],
            "failure_cause": failure_cause(evaluation) if dice < config.FAILURE_DICE_THRESHOLD else "",
        })
    return pd.DataFrame(rows)


def classifier_scores(predictions: pd.DataFrame) -> dict:
    """Accuracy, sensitivity, specificity and F1 of P(kidney) >= 0.5."""
    return metrics.binary_classification_scores(predictions["label"], (predictions["p_kidney"] >= 0.5).astype(int))


def summary_table(evaluations: dict, per_slice: dict, predictions: pd.DataFrame) -> pd.DataFrame:
    """Every phase's metrics in one table, train vs test, with mean / median / std."""
    rows = []
    for split in evaluations:  # "train" and "test", or only "test" when data/ is missing
        stage_names = list(evaluations[split][0]["phase1"].keys())
        for stage in stage_names:
            for metric in ("cnr", "psnr"):
                values = [e["phase1"][stage][metric] for e in evaluations[split]]
                if np.isnan(values).all():
                    continue
                rows.append({"phase": "1 enhancement", "metric": metric.upper(), "detail": stage, "split": split,
                             **statistics(values, "slices")})
        for strategy, column in (("region_growing", "dice"), ("multi_otsu", "dice_multi_otsu")):
            label = "B region growing + rule filter (FINAL)" if strategy == FINAL else "A multi-Otsu + rule filter"
            rows.append({"phase": "2 segmentation", "metric": "Dice", "detail": label, "split": split,
                         **statistics(per_slice[split][column], "slices")})
        rows.append({"phase": "2 segmentation", "metric": "IoU", "detail": "B region growing + rule filter (FINAL)",
                     "split": split, **statistics(per_slice[split]["iou"], "slices")})
        multi_otsu_iou = [metrics.iou_score(e["results"]["multi_otsu"]["final"], e["slice"]["ground_truth"])
                          for e in evaluations[split]]
        rows.append({"phase": "2 segmentation", "metric": "IoU", "detail": "A multi-Otsu + rule filter",
                     "split": split, **statistics(multi_otsu_iou, "slices")})

        # Phase 3: pooled over all candidates ("overall"), plus mean/median/std over patients.
        split_predictions = predictions[predictions["split"] == split]
        pooled = classifier_scores(split_predictions)
        per_patient = [classifier_scores(group) for _, group in split_predictions.groupby("case_id")]
        source = "nested out-of-fold" if split == "train" else "final model"
        for metric, key in (("accuracy", "accuracy"), ("sensitivity", "recall"), ("specificity", "specificity"),
                            ("F1", "f1")):
            rows.append({"phase": "3 classification", "metric": metric, "detail": f"k-NN, candidates ({source})",
                         "split": split, "overall": pooled[key],
                         **statistics([p[key] for p in per_patient], "patients")})
    table = pd.DataFrame(rows)
    return table[["phase", "metric", "detail", "split", "overall", "mean", "median", "std", "n", "unit"]]


def per_patient_table(per_slice_all: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: segmentation and classification results."""
    rows = []
    for (split, case_id), group in per_slice_all.groupby(["split", "case_id"], sort=False):
        scores = classifier_scores(predictions[predictions["case_id"] == case_id])
        rows.append({"case_id": case_id, "split": split, "n_slices": len(group),
                     "dice_mean": group["dice"].mean(), "dice_median": group["dice"].median(),
                     "dice_min": group["dice"].min(), "iou_mean": group["iou"].mean(),
                     "dice_multi_otsu_mean": group["dice_multi_otsu"].mean(),
                     "area_error_pct_mean": group["area_error_pct"].mean(),
                     "slices_below_threshold": int((group["dice"] < config.FAILURE_DICE_THRESHOLD).sum()),
                     "classifier_accuracy": scores["accuracy"], "classifier_f1": scores["f1"],
                     "tumour_slices": int(group["tumour_in_slice"].sum())})
    return pd.DataFrame(rows)


# =============================================================================
# 4. Figures
# =============================================================================
def figure_block_diagram(evaluation: dict) -> None:
    """Pipeline block diagram with the real intermediate images of one test slice."""
    slice_data, result, details = evaluation["slice"], evaluation["results"][FINAL], evaluation["details"]
    truth = slice_data["ground_truth"]
    dice = metrics.dice_coefficient(details["mask"], truth)
    window_low = config.HU_WINDOW_LEVEL - config.HU_WINDOW_WIDTH / 2
    blocks = [
        ("1  Raw CT slice", "Hounsfield Units\n(KiTS23)", slice_data["hu"]),
        ("2  Phase 1: enhancement", f"HU window [{window_low:g}, {window_low + config.HU_WINDOW_WIDTH:g}]\n"
                                    "Perona-Malik diffusion\nhomomorphic filter\nhistogram equalisation",
         evaluation["final_input"]),
        ("3  Phase 2: segmentation", "spine centroid (raw HU)\nseeds from learned offsets\nseeded region growing",
         evaluation["final_input"]),
        ("4  Morphology", "opening, closing\nhole filling (my SEs)\nsmall-object removal", result["candidates"]),
        ("5  Selection + classification", "rule filter: one region\nper side (learned zone)\n"
                                          "k-NN: P(kidney) per region", evaluation["windowed"]),
        ("6  Output", f"kidney mask\nred = predicted, green = truth\nDice {dice:.2f}", evaluation["windowed"]),
    ]
    figure, axes = plt.subplots(1, len(blocks), figsize=(4 * len(blocks), 6.2))
    for index, (axis, (title, text, image)) in enumerate(zip(axes, blocks)):
        axis.imshow(image, cmap="gray")
        if index == 2:
            axis.contour(result["raw"], levels=[0.5], colors="yellow", linewidths=0.8)
            for seed in result["seeds"]:
                if seed["seed"] is not None:
                    axis.plot(seed["seed"][1], seed["seed"][0], "o", color="orange", markersize=6)
            spine_row, spine_col = evaluation["anatomy"]["spine"]
            axis.plot(spine_col, spine_row, "c+", markersize=12, markeredgewidth=2)
        if index == 4:
            for region in details["regions"]:
                axis.contour(region["mask"], levels=[0.5], colors="cyan", linewidths=1)
                rows, cols = np.nonzero(region["mask"])
                axis.text(cols.mean(), rows.min() - 12, f"P={region['p_kidney']:.2f}", color="cyan",
                          ha="center", fontsize=9, fontweight="bold")
        if index == 5:
            axis.contour(truth, levels=[0.5], colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=1)
            axis.contour(details["mask"], levels=[0.5], colors=config.PREDICTED_CONTOUR_COLOUR, linewidths=1)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.text(0.5, -0.04, text, transform=axis.transAxes, ha="center", va="top", fontsize=9)
        axis.axis("off")
    figure.subplots_adjust(wspace=0.25, bottom=0.25)
    for left, right in zip(axes[:-1], axes[1:]):
        start = left.get_position()
        end = right.get_position()
        y = (start.y0 + start.y1) / 2
        figure.add_artist(FancyArrowPatch((start.x1 + 0.002, y), (end.x0 - 0.002, y), transform=figure.transFigure,
                                          arrowstyle="-|>", mutation_scale=18, color="black"))
    figure.suptitle(f"Kidney pipeline, end to end - test case {slice_data['case_id']} slice {slice_data['slice_index']}",
                    fontsize=13)
    save_figure(figure, "final_pipeline_block_diagram.png")


def figure_results_panel(test_evaluations: list[dict], test_table: pd.DataFrame) -> None:
    """Best, median and worst test slice of the final pipeline."""
    ordered = test_table.sort_values("dice").reset_index(drop=True)
    picks = {"best": ordered.iloc[-1], "median": ordered.iloc[len(ordered) // 2], "worst": ordered.iloc[0]}
    by_key = {(e["slice"]["case_id"], e["slice"]["slice_index"]): e for e in test_evaluations}
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.6), layout="constrained")
    for axis, (label, row) in zip(axes, picks.items()):
        evaluation = by_key[(row["case_id"], row["slice_index"])]
        plot_contour_overlay(axis, evaluation["windowed"], evaluation["details"]["mask"],
                             evaluation["slice"]["ground_truth"],
                             f"{label}: case {row['case_id']} slice {row['slice_index']} - Dice {row['dice']:.2f}")
        axis.title.set_fontsize(12)
    figure.suptitle("Final pipeline on TEST slices (red = predicted, green = ground truth)", fontsize=13)
    save_figure(figure, "final_test_results_panel.png")


def figure_dice_per_patient(per_slice_all: pd.DataFrame) -> None:
    """Dice of every slice, grouped by patient, training vs test."""
    patients = per_slice_all.groupby(["split", "case_id"], sort=False).size().reset_index()[["split", "case_id"]]
    patients = patients.sort_values(["split", "case_id"], ascending=[False, True]).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(11, 5))
    for position, row in patients.iterrows():
        dice = per_slice_all[(per_slice_all["case_id"] == row["case_id"])]["dice"]
        colour = "tab:blue" if row["split"] == "train" else "tab:red"
        axis.scatter(np.full(len(dice), position), dice, color=colour, alpha=0.7, s=30)
        axis.plot([position - 0.3, position + 0.3], [dice.mean()] * 2, color=colour, linewidth=3)
    axis.axhline(config.FAILURE_DICE_THRESHOLD, color="grey", linestyle=":",
                 label=f"failure threshold {config.FAILURE_DICE_THRESHOLD}")
    axis.set_xticks(range(len(patients)), [f"{c}\n{s}" for s, c in zip(patients["split"], patients["case_id"])])
    axis.set_ylabel("Dice")
    axis.set_ylim(0, 1.02)
    axis.set_title("Dice per patient - final pipeline (dots = slices, bar = patient mean; blue = train, red = test)")
    axis.legend(loc="lower left")
    save_figure(figure, "final_dice_per_patient.png")


def figure_failure_example(evaluation: dict, cause: str, dice: float) -> str:
    """One example slice for a failure cause, with the seeds shown. Returns the file name."""
    name = f"failure_{cause.replace(' ', '_')}.png"
    figure, axis = plt.subplots(figsize=(7, 7.4))
    slice_data = evaluation["slice"]
    plot_contour_overlay(axis, evaluation["windowed"], evaluation["details"]["mask"], slice_data["ground_truth"],
                         f"{cause}: case {slice_data['case_id']} slice {slice_data['slice_index']} - Dice {dice:.2f}\n"
                         "red = predicted, green = truth, yellow = grown region, orange = seed, cyan + = spine")
    axis.contour(evaluation["results"][FINAL]["raw"], levels=[0.5], colors="yellow", linewidths=0.8,
                 linestyles="dashed")
    for seed in evaluation["results"][FINAL]["seeds"]:
        if seed["seed"] is not None:
            axis.plot(seed["seed"][1], seed["seed"][0], "o", color="orange", markersize=7)
    spine_row, spine_col = evaluation["anatomy"]["spine"]
    axis.plot(spine_col, spine_row, "c+", markersize=14, markeredgewidth=2)
    save_figure(figure, name)
    return name


# =============================================================================
# 5. Failure analysis report
# =============================================================================
def write_failure_analysis(test_table: pd.DataFrame, test_evaluations: list[dict]) -> pd.DataFrame:
    """results/failure_analysis.md: test slices below the Dice threshold, grouped by cause."""
    failures = test_table[test_table["dice"] < config.FAILURE_DICE_THRESHOLD].sort_values("dice")
    by_key = {(e["slice"]["case_id"], e["slice"]["slice_index"]): e for e in test_evaluations}
    counts = failures["failure_cause"].value_counts()

    lines = ["# Failure analysis - final pipeline on the test set", "",
             f"Final pipeline: {FINAL.replace('_', ' ')} + rule filter, k-NN check "
             f"(`USE_CLASSIFIER_IN_PIPELINE = {config.USE_CLASSIFIER_IN_PIPELINE}`).",
             f"{len(failures)} of {len(test_table)} test slices have Dice below {config.FAILURE_DICE_THRESHOLD} "
             f"(test mean Dice {test_table['dice'].mean():.3f}, median {test_table['dice'].median():.3f}). "
             "Each is assigned ONE primary cause by fixed rules (scripts/step5_evaluate_test_set.py, "
             "`failure_cause`), checked in the order below.", "",
             "| Cause | Count | Slices (case/slice: Dice) |", "|---|---|---|"]
    for cause in FAILURE_CAUSES:
        rows = failures[failures["failure_cause"] == cause]
        listing = ", ".join(f"{r.case_id}/{r.slice_index}: {r.dice:.2f}" for r in rows.itertuples()) or "-"
        lines.append(f"| {cause} | {counts.get(cause, 0)} | {listing} |")
    lines.append("")

    for cause, (explanation, fix) in FAILURE_CAUSES.items():
        rows = failures[failures["failure_cause"] == cause]
        lines += [f"## {cause.capitalize()} ({len(rows)})", "", f"**What happens.** {explanation}", ""]
        if rows.empty:
            lines += ["No test slice failed for this reason.", ""]
        else:
            example = rows.iloc[0]
            figure_name = figure_failure_example(by_key[(example["case_id"], example["slice_index"])], cause,
                                                 example["dice"])
            lines += [f"![{cause}](figures/{figure_name})", "",
                      f"Example: case {example['case_id']} slice {example['slice_index']}, Dice {example['dice']:.2f}.", ""]
        lines += [f"**Realistic fix (future work).** {fix}", ""]

    config.FAILURE_ANALYSIS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"    saved {config.FAILURE_ANALYSIS_FILE.relative_to(config.PROJECT_ROOT)}")
    return failures


# =============================================================================
# 6. Export test images
# =============================================================================
def export_test_images(test_slices: list[dict]) -> None:
    """Copy the 24 test .npz files and write a PNG preview (soft-tissue window) of each."""
    folder = config.SAMPLE_TEST_IMAGES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for slice_data in test_slices:
        source = data_loading.slice_file_path(slice_data["case_id"], slice_data["slice_index"])
        if source.exists():  # in test-only mode the .npz files are already in this folder
            shutil.copy2(source, folder / source.name)
        plt.imsave(folder / source.with_suffix(".png").name, enh.apply_hu_window(slice_data["hu"]),
                   cmap="gray", vmin=0, vmax=255)
    (folder / "README.md").write_text(f"""# Sample test images

The {len(test_slices)} TEST slices used for the final evaluation (KiTS23 cases
{', '.join(str(c) for c in sorted({s['case_id'] for s in test_slices}))}; 6 slices each), included because
the results can be checked without downloading the full dataset.

- `case_XXXXX_slice_YYY.npz` - raw data: `hu` (int16 Hounsfield Units, not windowed),
  `kidney_mask` (1 = kidney, KiTS labels 1+2+3), `label_map` (KiTS labels 0-3),
  `pixel_spacing` (row_mm, col_mm), `case_id`, `slice_index`. Load with `numpy.load`.
- `case_XXXXX_slice_YYY.png` - preview only: soft-tissue window
  [{config.HU_WINDOW_LEVEL - config.HU_WINDOW_WIDTH / 2:g}, {config.HU_WINDOW_LEVEL + config.HU_WINDOW_WIDTH / 2:g}] HU
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
""", encoding="utf-8")
    print(f"    exported {len(test_slices)} test slices to {folder.relative_to(config.PROJECT_ROOT)}/")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print(f"Final pipeline: {FINAL} + {'k-NN' if config.USE_CLASSIFIER_IN_PIPELINE else 'rule filter'}, "
          f"input {config.SEGMENTATION_CONTRAST}; k-NN check with {len(config.KNN_FEATURES)} features, k = {config.KNN_K}")
    evaluations = {}
    for split in ("train", "test"):
        try:
            slices = data_loading.load_split(split)
        except FileNotFoundError:
            # A fresh copy of the repository has no data/: evaluate the test slices from
            # sample_test_images/ only (training rows are left out of the tables).
            print(f"   ({split} slices not available - data/ has not been prepared; test-only mode)")
            continue
        for slice_data in slices:
            slice_data["split"] = split
        print(f"1) Running the pipeline on {len(slices)} {split} slices...")
        evaluations[split] = [evaluate_slice(s) for s in slices]

    print("\n2) Tables")
    per_slice = {split: per_slice_table(evaluations[split]) for split in evaluations}
    predictions = pd.read_csv(config.TABLES_DIR / "step4_candidate_predictions.csv")
    summary = summary_table(evaluations, per_slice, predictions)
    save_table(summary, "final_results_summary.csv")
    save_table(per_slice["test"], "final_results_per_slice.csv")
    per_slice_all = pd.concat(list(per_slice.values()), ignore_index=True)
    save_table(per_patient_table(per_slice_all, predictions), "final_results_per_patient.csv")

    print("\n    Headline numbers (mean +/- std, median):")
    for _, row in summary.iterrows():
        if row["phase"].startswith("1") and not row["detail"].startswith(("0", "4")):
            continue
        overall = f"  overall {row['overall']:.3f}" if pd.notna(row["overall"]) else ""
        print(f"    {row['phase']:17s} {row['metric']:11s} {row['detail'][:40]:40s} {row['split']:5s} "
              f"{row['mean']:.3f} +/- {row['std']:.3f} (median {row['median']:.3f}){overall}")

    runtimes = per_slice["test"]["runtime_s"]
    print(f"\n    runtime per test slice: median {runtimes.median():.2f} s, mean {runtimes.mean():.2f} s, "
          f"max {runtimes.max():.2f} s (first call includes loading the models)")
    regions = sum(per_slice["test"]["n_regions"])
    low = sum(r["p_kidney"] < 0.5 for e in evaluations["test"] for r in e["details"]["regions"])
    print(f"    k-NN check on the final test regions: {regions} regions, {low} with P(kidney) < 0.5")

    print("\n3) Figures")
    by_key = {(e["slice"]["case_id"], e["slice"]["slice_index"]): e for e in evaluations["test"]}
    figure_block_diagram(by_key[config.EXAMPLE_HARD_TEST_SLICE])
    figure_results_panel(evaluations["test"], per_slice["test"])
    figure_dice_per_patient(per_slice_all)

    print("\n4) Failure analysis")
    failures = write_failure_analysis(per_slice["test"], evaluations["test"])
    for _, row in failures.iterrows():
        print(f"    case {row['case_id']} slice {row['slice_index']:3d}  Dice {row['dice']:.2f}  {row['failure_cause']}")

    print("\n5) Exporting test images")
    export_test_images([e["slice"] for e in evaluations["test"]])


if __name__ == "__main__":
    main()
