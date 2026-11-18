"""
Step 3 - compare the two Phase 2 segmentation strategies and choose their settings.

Run from the project folder:  .venv/bin/python scripts/step3_compare_segmentation.py

Every choice is made on the 8 TRAINING patients (48 slices), by mean Dice:
    1. learn priors from training ground truth (seed offsets, position/size zones,
       kidney grey-level range, region size cap) -> models/segmentation_priors.json
    2. Strategy A multi-Otsu: input (none / CLAHE / HE) x classes (3, 4, 5)
       x brightest classes kept (1, 2) x opening radius (2, 3, 5)
    3. Strategy B region growing: input x seed brightness (low percentile 10/40/60)
       x tolerance (10..50) x opening radius
    4. input comparison = best Dice per input and strategy (CLAHE vs HE vs none)
The settings found must be copied into config.py (the script prints them and warns
if config.py differs). Both strategies are then run with config.py on the training
AND the test patients; test numbers are reported separately and never used to choose.

Outputs:
    models/segmentation_priors.json
    results/tables/step3_input_comparison.csv
    results/tables/step3_otsu_classes_sweep.csv
    results/tables/step3_region_growing_sweep.csv
    results/tables/step3_segmentation_results.csv   (per slice, both strategies, train + test)
    results/tables/step3_summary.csv                (mean / median / std, train vs test)
    results/figures/step3_structuring_elements.png
    results/figures/step3_multi_otsu_class_map.png
    results/figures/step3_region_growing_seeds.png
    results/figures/step3_morphology_steps.png
    results/figures/step3_dice_boxplot.png
    results/figures/step3_best_test_cases.png
    results/figures/step3_worst_test_multi_otsu.png
    results/figures/step3_worst_test_region_growing.png
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to file without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import config  # noqa: E402
from src import data_loading, metrics  # noqa: E402
from src import phase1_enhancement as enh  # noqa: E402
from src import phase2_segmentation as seg  # noqa: E402
from src.pipeline import plot_contour_overlay  # noqa: E402

STRATEGIES = ["multi_otsu", "region_growing"]
STRATEGY_NAMES = {"multi_otsu": "A: multi-Otsu", "region_growing": "B: region growing"}
SIDE_NAMES = {"patient_right": "right", "patient_left": "left"}  # patient's sides


# =============================================================================
# Helpers
# =============================================================================
def prepare_slices(split: str) -> list[dict]:
    """Load a split and add landmarks, the windowed image and the three candidate inputs."""
    slices = data_loading.load_split(split)
    for slice_data in slices:
        slice_data["split"] = split
        slice_data["anatomy"] = seg.find_anatomy(slice_data["hu"])
        slice_data["windowed"] = enh.apply_hu_window(slice_data["hu"])
        before_contrast = enh.enhance_before_contrast(slice_data["hu"])
        slice_data["inputs"] = {method: enh.apply_contrast_step(before_contrast, method)
                                for method in config.SEGMENTATION_CONTRAST_CANDIDATES}
    return slices


def summarise_dice(rows: pd.DataFrame, group_columns: list[str], metric_columns=("dice",)) -> pd.DataFrame:
    """Mean, median and standard deviation of each metric over slices, one row per group."""
    summary = rows.groupby(group_columns, sort=False, dropna=False)[list(metric_columns)].agg(["mean", "median", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.insert(0, "n_slices", rows.groupby(group_columns, sort=False, dropna=False).size())
    return summary.reset_index()


def best_row(table: pd.DataFrame) -> pd.Series:
    """Row with the highest training Dice by config.SEGMENTATION_SELECTION_STATISTIC (mean)."""
    return table.loc[table[f"dice_{config.SEGMENTATION_SELECTION_STATISTIC}"].idxmax()]


def save_table(table: pd.DataFrame, name: str) -> None:
    """Write a CSV to results/tables/ with numbers rounded for readability."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(config.TABLES_DIR / name, index=False)
    print(f"    saved results/tables/{name}")


def save_figure(figure, name: str) -> None:
    """Write a PNG to results/figures/ and close it."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not figure.get_constrained_layout():
        figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(config.FIGURES_DIR / name, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"    saved results/figures/{name}")


def priors_with_seed_percentile(priors: dict, training_slices: list[dict], input_name: str,
                                low_percentile: float) -> dict:
    """Copy of the priors whose kidney grey-level range starts at a different low percentile."""
    adjusted = copy.deepcopy(priors)
    images = [s["inputs"][input_name] for s in training_slices]
    adjusted["intensity_range"][input_name] = seg.kidney_intensity_range(
        training_slices, images, (low_percentile, config.KIDNEY_INTENSITY_PERCENTILES[1]))
    return adjusted


# =============================================================================
# Training sweeps
# =============================================================================
def sweep_multi_otsu(training_slices: list[dict], priors: dict) -> pd.DataFrame:
    """Dice of every multi-Otsu setting on every training slice."""
    rows = []
    for input_name in config.SEGMENTATION_CONTRAST_CANDIDATES:
        for slice_data in training_slices:
            for n_classes in config.MULTI_OTSU_CLASS_CANDIDATES:
                for n_kidney_classes in config.MULTI_OTSU_KIDNEY_CLASS_CANDIDATES:
                    raw = seg.segment_multi_otsu_candidates(slice_data["inputs"][input_name], slice_data["anatomy"],
                                                            n_classes, n_kidney_classes)
                    for opening_radius in config.SE_OPENING_RADIUS_CANDIDATES:
                        final = seg.finish_segmentation(raw, slice_data["anatomy"], priors, opening_radius)["final"]
                        rows.append({"input": input_name, "n_classes": n_classes,
                                     "n_kidney_classes": n_kidney_classes, "opening_radius": opening_radius,
                                     "dice": metrics.dice_coefficient(final, slice_data["ground_truth"])})
    return summarise_dice(pd.DataFrame(rows), ["input", "n_classes", "n_kidney_classes", "opening_radius"])


def sweep_region_growing(training_slices: list[dict], priors: dict) -> pd.DataFrame:
    """Dice of every region-growing setting on every training slice."""
    rows = []
    for input_name in config.SEGMENTATION_CONTRAST_CANDIDATES:
        for low_percentile in config.KIDNEY_SEED_LOW_PERCENTILE_CANDIDATES:
            adjusted = priors_with_seed_percentile(priors, training_slices, input_name, low_percentile)
            for slice_data in training_slices:
                for tolerance in config.REGION_GROW_TOLERANCE_CANDIDATES:
                    raw, _ = seg.segment_region_growing_candidates(slice_data["inputs"][input_name],
                                                                   slice_data["anatomy"], adjusted, input_name,
                                                                   tolerance)
                    for opening_radius in config.SE_OPENING_RADIUS_CANDIDATES:
                        final = seg.finish_segmentation(raw, slice_data["anatomy"], adjusted, opening_radius)["final"]
                        rows.append({"input": input_name, "seed_low_percentile": low_percentile,
                                     "tolerance": tolerance, "opening_radius": opening_radius,
                                     "dice": metrics.dice_coefficient(final, slice_data["ground_truth"])})
    return summarise_dice(pd.DataFrame(rows), ["input", "seed_low_percentile", "tolerance", "opening_radius"])


def compare_inputs(otsu_table: pd.DataFrame, growing_table: pd.DataFrame) -> pd.DataFrame:
    """Best training Dice for each (input, strategy) pair - CLAHE vs HE compared by Dice."""
    rows = []
    for strategy, table, setting_columns in [
            ("multi_otsu", otsu_table, ["n_classes", "n_kidney_classes", "opening_radius"]),
            ("region_growing", growing_table, ["seed_low_percentile", "tolerance", "opening_radius"])]:
        for input_name in config.SEGMENTATION_CONTRAST_CANDIDATES:
            best = best_row(table[table["input"] == input_name])
            rows.append({"strategy": strategy, "input": input_name,
                         "best_setting": ", ".join(f"{c} {best[c]:g}" for c in setting_columns),
                         **{c: best[c] for c in ["dice_mean", "dice_median", "dice_std"]}})
    return pd.DataFrame(rows)


# =============================================================================
# Final evaluation with config.py (train and test)
# =============================================================================
def settings_from_config(strategy: str) -> dict:
    """The settings config.py holds for one strategy."""
    opening_radius = config.OPENING_RADIUS_FOR_STRATEGY[strategy]
    if strategy == "multi_otsu":
        return {"n_classes": config.MULTI_OTSU_CLASSES, "n_kidney_classes": config.MULTI_OTSU_KIDNEY_CLASSES,
                "opening_radius": opening_radius}
    return {"tolerance": config.REGION_GROW_TOLERANCE, "opening_radius": opening_radius}


def explain_failure(slice_data: dict, result: dict, strategy: str) -> str:
    """One-line, rule-based reason why a prediction disagrees with the ground truth.

    Checks, in order: nothing kept; each side's kidney missed (and why); a blob kept
    on a side with no kidney; leakage (low precision); under-segmentation (low recall,
    flagging tumour when most missed pixels are tumour); otherwise boundary error.
    """
    predicted, truth = result["final"], slice_data["ground_truth"]
    anatomy = slice_data["anatomy"]
    truth_by_side = seg.ground_truth_kidney_per_side(truth, anatomy)
    column_index = np.indices(truth.shape)[1]
    spine_col = anatomy["spine"][1]
    seed_status = {s["side"]: s for s in result["seeds"]}

    if not predicted.any():
        if strategy == "region_growing" and all(s["seed"] is None for s in result["seeds"]):
            return "nothing kept: no kidney-like seed found on either side"
        return "nothing kept: no blob inside the learned position/size zone"

    reasons = []
    for side, side_zone in zip(seg.SIDES, [column_index < spine_col, column_index >= spine_col]):
        name = SIDE_NAMES[side]
        predicted_side, truth_side = predicted & side_zone, truth_by_side[side]
        if truth_side is None:
            if predicted_side.sum() >= config.MIN_OBJECT_AREA_PX:
                reasons.append(f"false {name} kidney (only one kidney on this slice)")
            continue
        found_fraction = (predicted_side & truth_side).sum() / truth_side.sum()
        if found_fraction >= 0.1:
            continue
        if strategy == "region_growing":
            seed = seed_status[side]
            if seed["seed"] is None:
                reasons.append(f"missed {name} kidney: no kidney-like seed nearby")
            elif not truth_side[seed["seed"]]:
                reasons.append(f"missed {name} kidney: seed fell outside it")
            else:
                reasons.append(f"missed {name} kidney: region leaked, then rejected")
        else:
            in_candidates = (result["candidates"] & truth_side).sum() / truth_side.sum()
            if in_candidates >= 0.5:
                reasons.append(f"missed {name} kidney: blob merged/outside zone")
            else:
                reasons.append(f"missed {name} kidney: not in brightest class")
    if reasons:
        return "; ".join(reasons)

    overlap = (predicted & truth).sum()
    precision, recall = overlap / predicted.sum(), overlap / truth.sum()
    if precision < 0.75:
        return f"leaked into neighbouring tissue (precision {precision:.0%})"
    if recall < 0.75:
        missed = truth & ~predicted
        if (slice_data["label_map"][missed] == config.TUMOUR_LABEL).mean() > 0.5:
            return f"tumour not captured, different intensity (recall {recall:.0%})"
        return f"under-segmented (recall {recall:.0%})"
    return f"boundary disagreement (precision {precision:.0%}, recall {recall:.0%})"


def evaluate_with_config(slices: list[dict], priors: dict) -> tuple[pd.DataFrame, dict]:
    """Run both strategies with config.py settings; per-slice scores and the results for figures."""
    rows, results = [], {}
    for strategy in STRATEGIES:
        input_name = config.CONTRAST_FOR_STRATEGY[strategy]
        for slice_data in slices:
            result = seg.segment_slice(slice_data["inputs"][input_name], slice_data["anatomy"], priors,
                                       strategy, input_name, **settings_from_config(strategy))
            results[(strategy, slice_data["case_id"], slice_data["slice_index"])] = result
            predicted, truth = result["final"], slice_data["ground_truth"]
            without_morphology = seg.select_kidney_blobs_by_rules(result["raw"], slice_data["anatomy"], priors)
            overlap = (predicted & truth).sum()
            rows.append({
                "case_id": slice_data["case_id"], "slice_index": slice_data["slice_index"],
                "split": slice_data["split"], "strategy": strategy, "input": input_name,
                "dice": metrics.dice_coefficient(predicted, truth),
                "iou": metrics.iou_score(predicted, truth),
                "dice_without_morphology": metrics.dice_coefficient(without_morphology, truth),
                "precision": overlap / predicted.sum() if predicted.any() else 0.0,
                "recall": overlap / truth.sum(),
                "seeds": ", ".join(f"{SIDE_NAMES[s['side']]} {s['status']}" for s in result["seeds"]),
                "tumour_in_slice": bool((slice_data["label_map"] == config.TUMOUR_LABEL).any()),
                "reason": explain_failure(slice_data, result, strategy),
            })
    return pd.DataFrame(rows), results


# =============================================================================
# Figures
# =============================================================================
def figure_structuring_elements() -> None:
    """Show each hand-built structuring element as a grid of 0/1 values."""
    elements = {
        f"opening disk (r = {config.SE_OPENING_RADIUS})": seg.make_disk_structuring_element(config.SE_OPENING_RADIUS),
        f"closing disk (r = {config.SE_CLOSING_RADIUS})": seg.make_disk_structuring_element(config.SE_CLOSING_RADIUS),
        f"kidney ellipse {config.SE_KIDNEY_ELLIPSE_SIZE[0]}x{config.SE_KIDNEY_ELLIPSE_SIZE[1]}":
            seg.make_ellipse_structuring_element(*config.SE_KIDNEY_ELLIPSE_SIZE),
        f"bone-gap closing disk (r = {config.BONE_CLOSING_RADIUS})":
            seg.make_disk_structuring_element(config.BONE_CLOSING_RADIUS),
    }
    figure, axes = plt.subplots(1, len(elements), figsize=(4 * len(elements), 4.5))
    for axis, (name, element) in zip(axes, elements.items()):
        axis.imshow(element, cmap="Blues", vmin=0, vmax=1.4)
        for (row, col), value in np.ndenumerate(element):
            axis.text(col, row, str(value), ha="center", va="center", fontsize=7)
        axis.set_title(f"{name}\n{element.shape[0]}x{element.shape[1]}, {element.sum()} ones", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("My structuring elements (built from the circle / ellipse equation with numpy)")
    save_figure(figure, "step3_structuring_elements.png")


def figure_multi_otsu(slice_data: dict, priors: dict) -> None:
    """Input, class map, histogram with thresholds, candidates and final result for multi-Otsu."""
    input_name = config.CONTRAST_FOR_STRATEGY["multi_otsu"]
    image = slice_data["inputs"][input_name]
    region = slice_data["anatomy"]["body"] & ~slice_data["anatomy"]["bone_exclusion"]
    thresholds = seg.multi_otsu_thresholds(image, region, config.MULTI_OTSU_CLASSES)
    class_map = seg.multi_otsu_class_map(image, thresholds, region)
    result = seg.segment_slice(image, slice_data["anatomy"], priors, "multi_otsu", input_name,
                               **settings_from_config("multi_otsu"))
    dice = metrics.dice_coefficient(result["final"], slice_data["ground_truth"])

    figure, axes = plt.subplots(1, 4, figsize=(19, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"input ({input_name})", fontsize=9)
    shown = np.ma.masked_less(class_map, 0)
    class_image = axes[1].imshow(shown, cmap="viridis")
    figure.colorbar(class_image, ax=axes[1], ticks=range(config.MULTI_OTSU_CLASSES), fraction=0.046)
    axes[1].set_title(f"{config.MULTI_OTSU_CLASSES} classes (0 = darkest); kidney = brightest "
                      f"{config.MULTI_OTSU_KIDNEY_CLASSES}", fontsize=9)
    axes[2].hist(image[region], bins=64, range=(0, 255), color="grey")
    axes[2].hist(image[slice_data["ground_truth"]], bins=64, range=(0, 255), color="green", alpha=0.7)
    for threshold in thresholds:
        axes[2].axvline(threshold, color="red", linestyle="--")
    axes[2].set_title("histogram (body, no bone); green = kidney; red = thresholds", fontsize=9)
    axes[2].set_xlabel("grey level")
    axes[3].imshow(slice_data["windowed"], cmap="gray")
    axes[3].contour(result["candidates"], levels=[0.5], colors="yellow", linewidths=0.7)
    axes[3].contour(slice_data["ground_truth"], levels=[0.5], colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=1)
    axes[3].contour(result["final"], levels=[0.5], colors=config.PREDICTED_CONTOUR_COLOUR, linewidths=1)
    axes[3].set_title(f"yellow = all candidate blobs, red = kept, green = truth (Dice {dice:.2f})", fontsize=9)
    for axis in (axes[0], axes[1], axes[3]):
        axis.axis("off")
    figure.suptitle(f"Strategy A multi-Otsu - case {slice_data['case_id']} slice {slice_data['slice_index']} (training)")
    save_figure(figure, "step3_multi_otsu_class_map.png")


def figure_region_growing(examples: list[dict], priors: dict) -> None:
    """Spine, expected and accepted seeds, search radius, grown region and result."""
    input_name = config.CONTRAST_FOR_STRATEGY["region_growing"]
    figure, axes = plt.subplots(1, len(examples), figsize=(7 * len(examples), 7))
    for axis, slice_data in zip(np.atleast_1d(axes), examples):
        image = slice_data["inputs"][input_name]
        result = seg.segment_slice(image, slice_data["anatomy"], priors, "region_growing", input_name,
                                   **settings_from_config("region_growing"))
        dice = metrics.dice_coefficient(result["final"], slice_data["ground_truth"])
        axis.imshow(image, cmap="gray")
        axis.contour(result["raw"], levels=[0.5], colors="yellow", linewidths=0.8)
        axis.contour(slice_data["ground_truth"], levels=[0.5], colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=1)
        axis.contour(result["final"], levels=[0.5], colors=config.PREDICTED_CONTOUR_COLOUR, linewidths=1)
        spine_row, spine_col = slice_data["anatomy"]["spine"]
        axis.plot(spine_col, spine_row, "c+", markersize=16, markeredgewidth=2, label="spine centroid")
        radius = config.SEED_SEARCH_RADIUS_FRACTION * slice_data["anatomy"]["body_width"]
        for seed in result["seeds"]:
            expected_row, expected_col = seed["expected"]
            axis.plot(expected_col, expected_row, "mx", markersize=10, markeredgewidth=2)
            axis.add_patch(Circle((expected_col, expected_row), radius, fill=False, edgecolor="magenta", linestyle=":"))
            if seed["seed"] is not None:
                axis.plot(seed["seed"][1], seed["seed"][0], "o", color="orange", markersize=8)
            axis.annotate(f"{SIDE_NAMES[seed['side']]}: {seed['status']}", (expected_col, expected_row - radius - 4),
                          color="magenta", ha="center", fontsize=9)
        axis.set_title(f"case {slice_data['case_id']} slice {slice_data['slice_index']} - Dice {dice:.2f}\n"
                       "cyan + spine, magenta x expected seed (dotted = search radius), orange = seed used,\n"
                       "yellow = grown region, red = final, green = truth", fontsize=9)
        axis.axis("off")
    figure.suptitle(f"Strategy B seeded region growing (input {input_name}, tolerance {config.REGION_GROW_TOLERANCE})")
    save_figure(figure, "step3_region_growing_seeds.png")


def figure_morphology(slice_data: dict, priors: dict) -> None:
    """Each morphology step for the final strategy, with the Dice the rule filter would get at that step."""
    strategy = config.FINAL_SEGMENTATION_STRATEGY
    input_name = config.CONTRAST_FOR_STRATEGY[strategy]
    result = seg.segment_slice(slice_data["inputs"][input_name], slice_data["anatomy"], priors, strategy, input_name,
                               **settings_from_config(strategy))
    steps = {**result["steps"], "7 kidney blobs kept": result["final"]}
    figure, axes = plt.subplots(2, 4, figsize=(18, 10), layout="constrained")
    for axis, (name, mask) in zip(axes.ravel(), steps.items()):
        kept = mask if name.startswith("7") else seg.select_kidney_blobs_by_rules(mask, slice_data["anatomy"], priors)
        dice = metrics.dice_coefficient(kept, slice_data["ground_truth"])
        axis.imshow(mask, cmap="gray")
        axis.contour(slice_data["ground_truth"], levels=[0.5], colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=0.8)
        axis.set_title(f"{name}\n(Dice if kidney blobs picked now: {dice:.2f})", fontsize=9)
        axis.axis("off")
    axes.ravel()[-1].axis("off")
    figure.suptitle(f"Morphology steps - {STRATEGY_NAMES[strategy]}, case {slice_data['case_id']} slice "
                    f"{slice_data['slice_index']} (training). Green = ground truth.")
    save_figure(figure, "step3_morphology_steps.png")


def figure_dice_boxplot(per_slice: pd.DataFrame) -> None:
    """Dice distribution per strategy, training vs test."""
    figure, axis = plt.subplots(figsize=(8, 5))
    groups, labels = [], []
    for strategy in STRATEGIES:
        for split in ["train", "test"]:
            groups.append(per_slice[(per_slice["strategy"] == strategy) & (per_slice["split"] == split)]["dice"])
            labels.append(f"{STRATEGY_NAMES[strategy]}\n{split}")
    axis.boxplot(groups, tick_labels=labels, showmeans=True)
    axis.set_ylabel("Dice")
    axis.set_ylim(0, 1)
    axis.set_title("Dice per slice (triangle = mean)")
    save_figure(figure, "step3_dice_boxplot.png")


def figure_test_examples(per_slice: pd.DataFrame, slices_by_key: dict, results: dict, worst: bool) -> None:
    """Best test slices (one panel for both strategies) or the worst test slices (one figure per strategy)."""
    if not worst:
        figure, axes = plt.subplots(2, config.N_BEST_EXAMPLES, figsize=(5 * config.N_BEST_EXAMPLES, 10.5),
                                    layout="constrained")
        for row_number, strategy in enumerate(STRATEGIES):
            rows = per_slice[(per_slice["strategy"] == strategy) & (per_slice["split"] == "test")]
            for axis, (_, row) in zip(axes[row_number], rows.nlargest(config.N_BEST_EXAMPLES, "dice").iterrows()):
                key = (row["case_id"], row["slice_index"])
                plot_contour_overlay(axis, slices_by_key[key]["windowed"], results[(strategy, *key)]["final"],
                                     slices_by_key[key]["ground_truth"],
                                     f"{STRATEGY_NAMES[strategy]} - case {key[0]} slice {key[1]}\nDice {row['dice']:.2f}")
        figure.suptitle("Best TEST slices per strategy (red = predicted, green = ground truth)")
        save_figure(figure, "step3_best_test_cases.png")
        return

    for strategy in STRATEGIES:
        rows = per_slice[(per_slice["strategy"] == strategy) & (per_slice["split"] == "test")]
        worst_rows = rows.nsmallest(config.N_WORST_EXAMPLES, "dice")
        n_columns = 3
        n_rows = int(np.ceil(len(worst_rows) / n_columns))
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(5.5 * n_columns, 6.2 * n_rows), layout="constrained")
        for axis, (_, row) in zip(np.atleast_1d(axes).ravel(), worst_rows.iterrows()):
            key = (row["case_id"], row["slice_index"])
            plot_contour_overlay(axis, slices_by_key[key]["windowed"], results[(strategy, *key)]["final"],
                                 slices_by_key[key]["ground_truth"],
                                 f"case {key[0]} slice {key[1]} - Dice {row['dice']:.2f}\n{row['reason']}")
        figure.suptitle(f"{config.N_WORST_EXAMPLES} worst TEST slices - {STRATEGY_NAMES[strategy]} "
                        "(red = predicted, green = ground truth; left/right = patient's)")
        save_figure(figure, f"step3_worst_test_{strategy}.png")


# =============================================================================
# Main
# =============================================================================
def report_choice(name: str, chosen_on_training, value_in_config) -> None:
    """Print the training winner next to what config.py holds, flagging any mismatch."""
    status = "OK" if str(chosen_on_training) == str(value_in_config) else "<-- UPDATE config.py"
    print(f"    {name:40s} training winner: {str(chosen_on_training):22s} config.py: {str(value_in_config):22s} {status}")


def main() -> None:
    print("Preparing slices (landmarks + enhancement)...")
    train_slices = prepare_slices("train")
    test_slices = prepare_slices("test")
    print(f"Training slices: {len(train_slices)}   Test slices: {len(test_slices)} (test used only at the end)\n")

    # ---- 1. Priors from training ground truth ----
    print("1) Learning priors from training ground truth")
    priors = seg.learn_segmentation_priors(
        train_slices, {m: [s["inputs"][m] for s in train_slices] for m in config.SEGMENTATION_CONTRAST_CANDIDATES})
    seg.save_priors(priors)
    for side in seg.SIDES:
        side_priors = priors["sides"][side]
        print(f"    {SIDE_NAMES[side]:5s} kidney: seed offset dx {side_priors['seed_dx']:+.3f}, "
              f"dy {side_priors['seed_dy']:+.3f} (x body width), from {side_priors['n_training_kidneys']} kidneys")
    print(f"    saved {config.SEGMENTATION_PRIORS_FILE.relative_to(config.PROJECT_ROOT)}")

    # ---- 2-3. Sweeps ----
    print("\n2) Multi-Otsu sweep")
    otsu_table = sweep_multi_otsu(train_slices, priors)
    save_table(otsu_table, "step3_otsu_classes_sweep.csv")
    best_otsu = best_row(otsu_table)
    print(f"    best: input {best_otsu['input']}, {best_otsu['n_classes']:g} classes, brightest "
          f"{best_otsu['n_kidney_classes']:g}, opening r {best_otsu['opening_radius']:g} -> "
          f"Dice mean {best_otsu['dice_mean']:.3f}, median {best_otsu['dice_median']:.3f}")

    print("\n3) Region-growing sweep (takes a few minutes)")
    growing_table = sweep_region_growing(train_slices, priors)
    save_table(growing_table, "step3_region_growing_sweep.csv")
    best_growing = best_row(growing_table)
    print(f"    best: input {best_growing['input']}, seed range from p{best_growing['seed_low_percentile']:g}, "
          f"tolerance {best_growing['tolerance']:g}, opening r {best_growing['opening_radius']:g} -> "
          f"Dice mean {best_growing['dice_mean']:.3f}, median {best_growing['dice_median']:.3f}")

    # ---- 4. Input comparison ----
    print("\n4) Input comparison (best training Dice per input)")
    input_table = compare_inputs(otsu_table, growing_table)
    save_table(input_table, "step3_input_comparison.csv")
    print(input_table.round(3).to_string(index=False))

    final_strategy = "multi_otsu" if best_otsu["dice_mean"] > best_growing["dice_mean"] else "region_growing"
    print("\nChosen on training vs config.py:")
    report_choice("final strategy", final_strategy, config.FINAL_SEGMENTATION_STRATEGY)
    report_choice("contrast for multi-Otsu", best_otsu["input"], config.CONTRAST_FOR_STRATEGY["multi_otsu"])
    report_choice("contrast for region growing", best_growing["input"], config.CONTRAST_FOR_STRATEGY["region_growing"])
    report_choice("multi-Otsu (classes, brightest kept)", (int(best_otsu["n_classes"]), int(best_otsu["n_kidney_classes"])),
                  (config.MULTI_OTSU_CLASSES, config.MULTI_OTSU_KIDNEY_CLASSES))
    report_choice("region growing (seed low pct, tolerance)",
                  (int(best_growing["seed_low_percentile"]), int(best_growing["tolerance"])),
                  (int(config.KIDNEY_INTENSITY_PERCENTILES[0]), int(config.REGION_GROW_TOLERANCE)))
    report_choice("opening radius multi-Otsu", int(best_otsu["opening_radius"]),
                  config.OPENING_RADIUS_FOR_STRATEGY["multi_otsu"])
    report_choice("opening radius region growing", int(best_growing["opening_radius"]),
                  config.OPENING_RADIUS_FOR_STRATEGY["region_growing"])

    # ---- 5. Both strategies with config.py, train and test ----
    print("\n5) Evaluation with config.py settings")
    all_slices = train_slices + test_slices
    per_slice, results = evaluate_with_config(all_slices, priors)
    save_table(per_slice, "step3_segmentation_results.csv")
    summary = summarise_dice(per_slice, ["split", "strategy"], ("dice", "iou", "dice_without_morphology"))
    save_table(summary, "step3_summary.csv")
    print(summary[["split", "strategy", "dice_mean", "dice_median", "dice_std", "iou_mean",
                   "dice_without_morphology_mean"]].round(3).to_string(index=False))
    print("\n    Worst TEST slices:")
    for strategy in STRATEGIES:
        rows = per_slice[(per_slice["strategy"] == strategy) & (per_slice["split"] == "test")]
        for _, row in rows.nsmallest(config.N_WORST_EXAMPLES, "dice").iterrows():
            print(f"    {strategy:15s} case {row['case_id']} slice {row['slice_index']:3d}  "
                  f"Dice {row['dice']:.2f}  {row['reason']}")

    # ---- 6. Figures ----
    print("\n6) Figures")
    slices_by_key = {(s["case_id"], s["slice_index"]): s for s in all_slices}
    example = slices_by_key[config.EXAMPLE_TRAIN_SLICE]
    figure_structuring_elements()
    figure_multi_otsu(example, priors)
    # Second region-growing example: a training slice where a seed had to be skipped (one kidney only).
    training_growing = per_slice[(per_slice["strategy"] == "region_growing") & (per_slice["split"] == "train")]
    skipped = training_growing[training_growing["seeds"].str.contains("skipped")]
    second_key = tuple(skipped.iloc[0][["case_id", "slice_index"]]) if not skipped.empty else (17, 23)
    figure_region_growing([example, slices_by_key[second_key]], priors)
    figure_morphology(example, priors)
    figure_dice_boxplot(per_slice)
    figure_test_examples(per_slice, slices_by_key, results, worst=False)
    figure_test_examples(per_slice, slices_by_key, results, worst=True)


if __name__ == "__main__":
    main()
