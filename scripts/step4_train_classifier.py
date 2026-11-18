"""
Step 4 - Phase 3: features + k-NN, trained and tuned on the TRAINING patients only.

Run from the project folder:  .venv/bin/python scripts/step4_train_classifier.py
(needs models/segmentation_priors.json from step 3)

What it does:
    1. candidates on every slice = all multi-Otsu blobs + every region-growing region
       (up to SEED_MAX_ATTEMPTS seeds per side), labelled kidney if >= 50% inside the truth
    2. 17 features per candidate
    3. feature selection: drop compactness (= 4*pi / circularity), then backward
       selection on grouped 5-fold CV F1
    4. k (1..25, odd) and weighting (uniform / distance) retuned on the selected features
       by 5-fold cross-validation grouped by patient, optimising F1; scaler + k-NN saved
    5. decision, on TRAINING Dice only: does the k-NN pipeline beat the rule filter?
       Training Dice and the training classifier scores use NESTED cross-validation:
       feature selection and k tuning are repeated inside each outer fold, so a
       held-out patient never influences the model that classifies it.
    4. classifier scored on the TEST candidates (reported separately)
    5. Dice of both strategies before (Phase 2 rule filter) and after the k-NN.
       Training slices are classified by NESTED OUT-OF-FOLD models (features, k and
       weights all chosen without that patient), so training Dice is not inflated.

Outputs:
    models/feature_scaler.joblib, models/knn_classifier.joblib
    results/tables/step4_candidate_counts.csv
    results/tables/step4_feature_selection.csv
    results/tables/step4_nested_selection.csv
    results/tables/step4_cv_results.csv
    results/tables/step4_classifier_test_results.csv
    results/tables/step4_candidate_predictions.csv
    results/tables/step4_feature_analysis.csv
    results/tables/step4_dice_before_after_knn.csv
    results/tables/step4_dice_per_slice.csv
    results/tables/step4_training_features.csv
    results/figures/step4_confusion_matrix_test.png
    results/figures/step4_feature_selection.png
    results/figures/step4_nested_selection_stability.png
    results/figures/step4_cv_f1_vs_k.png
    results/figures/step4_feature_distributions.png
    results/figures/step4_candidates_knn_decisions.png
    results/figures/step4_seed_retry_examples.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to file without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import config  # noqa: E402
from src import data_loading, metrics  # noqa: E402
from src import phase1_enhancement as enh  # noqa: E402
from src import phase2_segmentation as seg  # noqa: E402
from src import phase3_classification as clf  # noqa: E402
from src.pipeline import plot_contour_overlay  # noqa: E402

STRATEGIES = ["multi_otsu", "region_growing"]
METHOD_NAMES = {
    ("multi_otsu", "rule filter"): "A multi-Otsu + rule filter (step 3)",
    ("multi_otsu", "k-NN"): "A multi-Otsu + k-NN",
    ("region_growing", "rule filter"): "B region growing + rule filter (step 3)",
    ("region_growing", "k-NN, first seed only"): "B region growing + k-NN, 1 seed",
    ("region_growing", "k-NN + seed retry"): f"B region growing + k-NN, up to {config.SEED_MAX_ATTEMPTS} seeds",
}


# =============================================================================
# Helpers
# =============================================================================
def prepare_slices(split: str) -> list[dict]:
    """Load a split and add landmarks, the windowed image and each strategy's input image."""
    slices = data_loading.load_split(split)
    for slice_data in slices:
        slice_data["split"] = split
        slice_data["anatomy"] = seg.find_anatomy(slice_data["hu"])
        slice_data["windowed"] = enh.apply_hu_window(slice_data["hu"])
        before_contrast = enh.enhance_before_contrast(slice_data["hu"])
        slice_data["inputs"] = {strategy: enh.apply_contrast_step(before_contrast, config.CONTRAST_FOR_STRATEGY[strategy])
                                for strategy in STRATEGIES}
    return slices


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


# =============================================================================
# 1-2. Candidates, labels and features
# =============================================================================
def candidates_for_slice(slice_data: dict, priors: dict) -> pd.DataFrame:
    """All candidates of one slice (multi-Otsu blobs + region-growing regions) with features and labels."""
    blobs, sources = [], []
    for blob in clf.multi_otsu_candidate_blobs(slice_data["inputs"]["multi_otsu"], slice_data["anatomy"]):
        blobs.append(blob)
        sources.append(("multi_otsu", None))
    for side in seg.SIDES:
        # No classifier: every seed attempt is run, so rejected-looking regions become negatives too.
        attempts = clf.grow_side_with_retries(slice_data["inputs"]["region_growing"], slice_data["hu"],
                                              slice_data["windowed"], slice_data["anatomy"], priors,
                                              config.CONTRAST_FOR_STRATEGY["region_growing"], side)
        for attempt_number, attempt in enumerate(attempts, start=1):
            for blob in attempt["blobs"]:
                blobs.append(blob)
                sources.append(("region_growing", attempt_number))

    features = clf.features_for_blobs(blobs, slice_data["hu"], slice_data["windowed"], slice_data["anatomy"])
    table = pd.DataFrame(features, columns=clf.FEATURE_NAMES)
    table.insert(0, "case_id", slice_data["case_id"])
    table.insert(1, "slice_index", slice_data["slice_index"])
    table.insert(2, "split", slice_data["split"])
    table.insert(3, "source", [source for source, _ in sources])
    table.insert(4, "seed_attempt", [attempt for _, attempt in sources])
    table["label"] = [clf.label_blob_against_ground_truth(blob, slice_data["ground_truth"]) for blob in blobs]
    return table


def build_candidate_table(slices: list[dict], priors: dict) -> pd.DataFrame:
    """Candidates of every slice in one table."""
    return pd.concat([candidates_for_slice(s, priors) for s in slices], ignore_index=True)


def feature_analysis(training_table: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each feature for kidney vs not-kidney (training), and how well it separates them.

    Separation = |Cohen's d| = |mean(kidney) - mean(not kidney)| / pooled standard deviation.
    About 0.2 = small, 0.5 = medium, 0.8+ = large difference between the classes.
    """
    rows = []
    kidney = training_table[training_table["label"] == 1]
    other = training_table[training_table["label"] == 0]
    for name in clf.FEATURE_NAMES:
        pooled_std = np.sqrt((kidney[name].var() * (len(kidney) - 1) + other[name].var() * (len(other) - 1))
                             / (len(kidney) + len(other) - 2))
        rows.append({"feature": name,
                     "kidney_mean": kidney[name].mean(), "kidney_std": kidney[name].std(),
                     "not_kidney_mean": other[name].mean(), "not_kidney_std": other[name].std(),
                     "separation_abs_cohens_d": abs(kidney[name].mean() - other[name].mean()) / pooled_std})
    return pd.DataFrame(rows).sort_values("separation_abs_cohens_d", ascending=False).reset_index(drop=True)


# =============================================================================
# 3. Out-of-fold models for honest training Dice
# =============================================================================
def train_nested_out_of_fold_models(training_table: pd.DataFrame) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Nested cross-validation: for each outer fold, choose features AND k using only that fold's
    training patients, then classify the held-out patients.

    Outer loop: 5 folds, each holding out 1-2 whole patients.
    Inside each fold (the other 6-7 patients only):
        1. backward feature selection with its own patient-grouped 5-fold CV
        2. retune k and weighting on the selected features, same inner CV
        3. fit scaler + k-NN on those patients
    So a held-out patient never influences which features or which k are used to
    classify it, and the out-of-fold scores (and training Dice) are not optimistic.

    Outputs:
        {case_id: (scaler, knn) to use for that patient}, the out-of-fold P(kidney) for
        every training candidate, and a log table (one row per outer fold).
    """
    labels = training_table["label"].to_numpy()
    groups = training_table["case_id"].to_numpy()
    starting = [f for f in clf.FEATURE_NAMES if f not in config.KNN_FEATURES_DROPPED_BEFORE_SELECTION]
    model_for_case, out_of_fold, log = {}, np.zeros(len(training_table)), []

    outer_folds = GroupKFold(n_splits=config.CV_FOLDS).split(training_table, labels, groups)
    for fold_number, (train_index, held_out_index) in enumerate(outer_folds, start=1):
        fold_table = training_table.iloc[train_index].reset_index(drop=True)
        fold_labels, fold_groups = labels[train_index], groups[train_index]

        selected, _ = clf.backward_feature_selection(fold_table, fold_labels, fold_groups, starting,
                                                     config.FEATURE_SELECTION_K, config.FEATURE_SELECTION_WEIGHTS)
        best, cv_table = clf.tune_k_by_grouped_cross_validation(fold_table[selected], fold_labels, fold_groups)
        scaler = clf.fit_feature_scaler(fold_table[selected])
        knn = clf.train_knn_classifier(scaler.transform(fold_table[selected]), fold_labels, best["k"], best["weights"])

        out_of_fold[held_out_index] = clf.predict_kidney_probability(scaler, knn, training_table.iloc[held_out_index])
        held_out_cases = sorted(int(c) for c in np.unique(groups[held_out_index]))
        for case_id in held_out_cases:
            model_for_case[case_id] = (scaler, knn)
        inner = cv_table[(cv_table["k"] == best["k"]) & (cv_table["weights"] == best["weights"])].iloc[0]
        log.append({"outer_fold": fold_number, "held_out_cases": " ".join(map(str, held_out_cases)),
                    "n_features": len(selected), "selected_features": ", ".join(selected),
                    "k": best["k"], "weights": best["weights"], "inner_cv_f1": inner["f1_mean"],
                    **{f"uses_{name}": name in selected for name in clf.FEATURE_NAMES}})
    return model_for_case, out_of_fold, pd.DataFrame(log)


# =============================================================================
# 5. Dice before and after the k-NN
# =============================================================================
def dice_before_after(slices: list[dict], priors: dict, model_for_slice) -> tuple[pd.DataFrame, dict]:
    """Every strategy/method on every slice. model_for_slice(slice) -> (scaler, knn) to use."""
    rows, kept = [], {}
    for slice_data in slices:
        scaler, knn = model_for_slice(slice_data)
        key = (slice_data["case_id"], slice_data["slice_index"])
        masks = {}
        for strategy in STRATEGIES:
            image, input_name = slice_data["inputs"][strategy], config.CONTRAST_FOR_STRATEGY[strategy]
            settings = {"opening_radius": config.OPENING_RADIUS_FOR_STRATEGY[strategy]}
            if strategy == "multi_otsu":
                settings.update(n_classes=config.MULTI_OTSU_CLASSES, n_kidney_classes=config.MULTI_OTSU_KIDNEY_CLASSES)
            masks[(strategy, "rule filter")] = seg.segment_slice(image, slice_data["anatomy"], priors, strategy,
                                                                 input_name, **settings)["final"]
            if strategy == "multi_otsu":
                result = clf.segment_with_knn(slice_data["hu"], image, slice_data["anatomy"], priors, strategy,
                                              input_name, scaler, knn)
                masks[(strategy, "k-NN")] = result["final"]
                kept[("knn_otsu", *key)] = result
            else:
                first_only = clf.segment_with_knn(slice_data["hu"], image, slice_data["anatomy"], priors, strategy,
                                                  input_name, scaler, knn, max_attempts=1)
                with_retry = clf.segment_with_knn(slice_data["hu"], image, slice_data["anatomy"], priors, strategy,
                                                  input_name, scaler, knn)
                masks[(strategy, "k-NN, first seed only")] = first_only["final"]
                masks[(strategy, "k-NN + seed retry")] = with_retry["final"]
                kept[("retry", *key)] = with_retry
        for (strategy, method), mask in masks.items():
            kept[(strategy, method, *key)] = mask
            rows.append({"case_id": key[0], "slice_index": key[1], "split": slice_data["split"],
                         "strategy": strategy, "method": method,
                         "dice": metrics.dice_coefficient(mask, slice_data["ground_truth"]),
                         "iou": metrics.iou_score(mask, slice_data["ground_truth"])})
    return pd.DataFrame(rows), kept


# =============================================================================
# Figures
# =============================================================================
def figure_confusion_matrix(scores: dict) -> None:
    """Test confusion matrix with counts."""
    matrix = np.array([[scores["true_negatives"], scores["false_positives"]],
                       [scores["false_negatives"], scores["true_positives"]]])
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        axis.text(col, row, str(value), ha="center", va="center", fontsize=16,
                  color="white" if value > matrix.max() / 2 else "black")
    axis.set_xticks([0, 1], ["not kidney", "kidney"])
    axis.set_yticks([0, 1], ["not kidney", "kidney"])
    axis.set_xlabel("predicted by k-NN")
    axis.set_ylabel("true (from ground truth)")
    axis.set_title(f"TEST candidates (k = {config.KNN_K}, {config.KNN_WEIGHTS})\n"
                   f"accuracy {scores['accuracy']:.3f}, recall {scores['recall']:.3f}, "
                   f"specificity {scores['specificity']:.3f}, F1 {scores['f1']:.3f}", fontsize=9)
    save_figure(figure, "step4_confusion_matrix_test.png")


def figure_feature_selection(log: pd.DataFrame) -> None:
    """CV F1 of the feature set after each backward-selection round."""
    starting = log[log["round"] <= 0]
    removed = log[log["removed"]]
    labels = list(starting["tried_removing"]) + [f"- {name}" for name in removed["tried_removing"]]
    f1_mean = list(starting["f1_mean"]) + list(removed["f1_mean"])
    f1_std = list(starting["f1_std"]) + list(removed["f1_std"])
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(range(len(labels)), f1_mean, yerr=f1_std, fmt="o-", capsize=4)
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_ylabel("grouped 5-fold CV F1 (± std over folds)")
    axis.set_title(f"Backward feature selection on training candidates (k = {config.FEATURE_SELECTION_K}, "
                   f"{config.FEATURE_SELECTION_WEIGHTS}). Each step removes the feature whose removal helps most.",
                   fontsize=9)
    figure.tight_layout()
    save_figure(figure, "step4_feature_selection.png")


def figure_nested_selection(fold_log: pd.DataFrame, final_features: list[str]) -> None:
    """Which features each outer fold selected, next to the final selection on all 8 patients."""
    columns = [f"fold {row['outer_fold']}\n(held out {row['held_out_cases']})" for _, row in fold_log.iterrows()]
    columns.append("final model\n(all 8 patients)")
    grid = np.array([[bool(row[f"uses_{name}"]) for _, row in fold_log.iterrows()] + [name in final_features]
                     for name in clf.FEATURE_NAMES], dtype=float)
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(grid, cmap="Greens", vmin=0, vmax=1.3, aspect="auto")
    axis.set_xticks(range(len(columns)), columns, fontsize=8)
    counts = grid[:, :-1].sum(axis=1).astype(int)
    axis.set_yticks(range(len(clf.FEATURE_NAMES)),
                    [f"{name} ({count}/{len(fold_log)})" for name, count in zip(clf.FEATURE_NAMES, counts)], fontsize=8)
    for (row, col), value in np.ndenumerate(grid):
        if value:
            axis.text(col, row, "selected", ha="center", va="center", fontsize=7)
    axis.set_title("Nested feature selection: features chosen inside each outer fold "
                   "(number = how many of the 5 folds chose it)", fontsize=9)
    figure.tight_layout()
    save_figure(figure, "step4_nested_selection_stability.png")


def figure_cv_curve(cv_table: pd.DataFrame, best: dict) -> None:
    """Mean CV F1 (± std) against k for both weightings."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for weights, colour in [("uniform", "tab:blue"), ("distance", "tab:orange")]:
        rows = cv_table[cv_table["weights"] == weights]
        axis.plot(rows["k"], rows["f1_mean"], "o-", color=colour, label=f"{weights} weights")
        axis.fill_between(rows["k"], rows["f1_mean"] - rows["f1_std"], rows["f1_mean"] + rows["f1_std"],
                          color=colour, alpha=0.15)
    axis.axvline(best["k"], color="grey", linestyle="--", label=f"chosen k = {best['k']} ({best['weights']})")
    axis.set_xlabel("k (number of neighbours)")
    axis.set_ylabel("F1 (mean over 5 patient-grouped folds, band = ± std)")
    axis.set_xticks(config.KNN_K_CANDIDATES)
    axis.legend()
    axis.set_title("Cross-validation on training patients")
    save_figure(figure, "step4_cv_f1_vs_k.png")


def figure_feature_distributions(training_table: pd.DataFrame, analysis: pd.DataFrame) -> None:
    """Box plots of each feature, kidney vs not-kidney (training), ordered by separation."""
    figure, axes = plt.subplots(3, 6, figsize=(22, 11))
    for axis, (_, row) in zip(axes.ravel(), analysis.iterrows()):
        name = row["feature"]
        groups = [training_table.loc[training_table["label"] == label, name] for label in (0, 1)]
        boxes = axis.boxplot(groups, tick_labels=["not kidney", "kidney"], showfliers=False, patch_artist=True)
        for patch, colour in zip(boxes["boxes"], ["tab:orange", "tab:green"]):
            patch.set_facecolor(colour)
        axis.set_title(f"{name}\n|d| = {row['separation_abs_cohens_d']:.2f}", fontsize=9)
    axes.ravel()[-1].axis("off")
    figure.suptitle("Feature distributions on TRAINING candidates, best-separating first "
                    "(|d| = |Cohen's d|; outliers hidden)")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, "step4_feature_distributions.png")


def figure_candidate_decisions(test_slices: list[dict], per_slice: pd.DataFrame, kept: dict) -> None:
    """Multi-Otsu candidates on 3 test slices (best, middle, worst Dice), coloured by the k-NN decision."""
    rows = per_slice[(per_slice["split"] == "test") & (per_slice["method"] == "k-NN")].sort_values("dice")
    picks = [rows.iloc[-1], rows.iloc[len(rows) // 2], rows.iloc[0]]
    by_key = {(s["case_id"], s["slice_index"]): s for s in test_slices}
    figure, axes = plt.subplots(1, 3, figsize=(20, 7.5), layout="constrained")
    for axis, row, label in zip(axes, picks, ["best", "middle", "worst"]):
        key = (row["case_id"], row["slice_index"])
        slice_data, result = by_key[key], kept[("knn_otsu", *key)]
        axis.imshow(slice_data["windowed"], cmap="gray")
        axis.contour(slice_data["ground_truth"], levels=[0.5], colors=config.GROUND_TRUTH_CONTOUR_COLOUR, linewidths=1.2)
        for blob, probability in zip(result["blobs"], result["probabilities"]):
            colour = "cyan" if probability >= 0.5 else "orange"
            axis.contour(blob, levels=[0.5], colors=colour, linewidths=1)
            rows_, cols_ = np.nonzero(blob)
            axis.text(cols_.mean(), rows_.mean(), f"{probability:.2f}", color=colour, fontsize=7,
                      ha="center", va="center", fontweight="bold")
        axis.contour(result["final"], levels=[0.5], colors=config.PREDICTED_CONTOUR_COLOUR, linewidths=1.6)
        axis.set_title(f"{label}: case {key[0]} slice {key[1]} - Dice {row['dice']:.2f}", fontsize=10)
        axis.axis("off")
    figure.suptitle("Multi-Otsu candidates classified by the k-NN (TEST). Cyan = kidney (P >= 0.5), "
                    "orange = not kidney, number = P(kidney); red = kept (best per side), green = ground truth")
    save_figure(figure, "step4_candidates_knn_decisions.png")


def figure_seed_retry(all_slices: list[dict], per_slice: pd.DataFrame, kept: dict) -> None:
    """Slices where trying the next seed fixed a failure: first-seed result vs result with retries."""
    rows = per_slice[per_slice["strategy"] == "region_growing"].pivot_table(
        index=["case_id", "slice_index", "split"], columns="method", values="dice").reset_index()
    rows["gain"] = rows["k-NN + seed retry"] - rows["k-NN, first seed only"]
    # Prefer test examples, then training, largest gains first.
    rows = rows[rows["gain"] > 0.05].sort_values(["split", "gain"], ascending=[True, False]).head(3)
    if rows.empty:
        print("    no slice where the seed retry improved Dice by > 0.05 - retry figure skipped")
        return
    by_key = {(s["case_id"], s["slice_index"]): s for s in all_slices}
    figure, axes = plt.subplots(len(rows), 2, figsize=(13, 6.5 * len(rows)), layout="constrained", squeeze=False)
    for axes_row, (_, row) in zip(axes, rows.iterrows()):
        key = (int(row["case_id"]), int(row["slice_index"]))
        slice_data = by_key[key]
        plot_contour_overlay(axes_row[0], slice_data["windowed"],
                             kept[("region_growing", "k-NN, first seed only", *key)], slice_data["ground_truth"],
                             f"{row['split']} case {key[0]} slice {key[1]}: first seed only - "
                             f"Dice {row['k-NN, first seed only']:.2f}")
        plot_contour_overlay(axes_row[1], slice_data["windowed"],
                             kept[("region_growing", "k-NN + seed retry", *key)], slice_data["ground_truth"],
                             f"with seed retry - Dice {row['k-NN + seed retry']:.2f}\n"
                             "numbers = seed attempt (x = rejected by k-NN, o = accepted)")
        for side, attempts in kept[("retry", *key)]["attempts"].items():
            for number, attempt in enumerate(attempts, start=1):
                accepted = attempt["accepted"] is not None
                axes_row[1].plot(attempt["seed"][1], attempt["seed"][0], "o" if accepted else "x",
                                 color="yellow" if accepted else "magenta", markersize=9, markeredgewidth=2)
                axes_row[1].annotate(str(number), (attempt["seed"][1] + 6, attempt["seed"][0] - 6),
                                     color="yellow", fontsize=10, fontweight="bold")
    figure.suptitle("Seed retry: when the k-NN rejects a grown region, the next-best seed on that side is tried "
                    "(red = predicted, green = ground truth)")
    save_figure(figure, "step4_seed_retry_examples.png")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    priors = seg.load_priors()
    print("Preparing slices...")
    train_slices, test_slices = prepare_slices("train"), prepare_slices("test")

    # ---- 1-2. Candidates and features ----
    print("\n1) Candidates and 17 features")
    candidates = build_candidate_table(train_slices + test_slices, priors)
    training_table = candidates[candidates["split"] == "train"].reset_index(drop=True)
    test_table = candidates[candidates["split"] == "test"].reset_index(drop=True)
    counts = candidates.groupby(["split", "source", "label"]).size().unstack("label", fill_value=0)
    counts.columns = ["not_kidney", "kidney"]
    counts["total"] = counts.sum(axis=1)
    counts = counts.reset_index()
    save_table(counts, "step4_candidate_counts.csv")
    save_table(training_table, "step4_training_features.csv")
    print(counts.to_string(index=False))

    labels_train = training_table["label"].to_numpy()
    groups_train = training_table["case_id"].to_numpy()

    # ---- Feature selection (training only) ----
    print(f"\n2) Feature selection (k = {config.FEATURE_SELECTION_K}, {config.FEATURE_SELECTION_WEIGHTS}; "
          "grouped 5-fold CV F1)")
    all_17 = clf.grouped_cv_scores(training_table[clf.FEATURE_NAMES], labels_train, groups_train,
                                   config.FEATURE_SELECTION_K, config.FEATURE_SELECTION_WEIGHTS)
    starting = [f for f in clf.FEATURE_NAMES if f not in config.KNN_FEATURES_DROPPED_BEFORE_SELECTION]
    selected, selection_log = clf.backward_feature_selection(training_table, labels_train, groups_train, starting,
                                                             config.FEATURE_SELECTION_K,
                                                             config.FEATURE_SELECTION_WEIGHTS)
    baseline_row = pd.DataFrame([{"round": -1, "tried_removing": "(all 17 features)", "n_features": 17,
                                  **all_17, "removed": False}])
    selection_log = pd.concat([baseline_row, selection_log], ignore_index=True)
    selection_log.loc[selection_log["round"] == 0, "tried_removing"] = \
        f"(start: 17 - {', '.join(config.KNN_FEATURES_DROPPED_BEFORE_SELECTION)})"
    save_table(selection_log, "step4_feature_selection.csv")
    print(f"    all 17 features:          CV F1 {all_17['f1_mean']:.3f} +/- {all_17['f1_std']:.3f}")
    start_row = selection_log[selection_log["round"] == 0].iloc[0]
    print(f"    without compactness (16): CV F1 {start_row['f1_mean']:.3f} +/- {start_row['f1_std']:.3f}")
    for _, row in selection_log[selection_log["removed"]].iterrows():
        print(f"    round {row['round']}: removed {row['tried_removing']:18s} -> CV F1 {row['f1_mean']:.3f} "
              f"± {row['f1_std']:.3f} ({row['n_features']} features)")
    print(f"    selected ({len(selected)}): {selected}")
    status = "OK" if selected == config.KNN_FEATURES else "<-- UPDATE config.KNN_FEATURES"
    print(f"    config.py KNN_FEATURES matches: {status}")

    # ---- 3. Cross-validation (training only) ----
    print("\n3) Retune k x weighting on the selected features (5 folds grouped by patient, F1)")
    best, cv_table = clf.tune_k_by_grouped_cross_validation(training_table[selected], labels_train, groups_train)
    save_table(cv_table, "step4_cv_results.csv")
    best_row = cv_table[(cv_table["k"] == best["k"]) & (cv_table["weights"] == best["weights"])].iloc[0]
    print(f"    best: k = {best['k']}, {best['weights']} weights -> CV F1 {best_row['f1_mean']:.3f} "
          f"± {best_row['f1_std']:.3f}, accuracy {best_row['accuracy_mean']:.3f} ± {best_row['accuracy_std']:.3f}")
    status = "OK" if (best["k"], best["weights"]) == (config.KNN_K, config.KNN_WEIGHTS) else "<-- UPDATE config.py"
    print(f"    config.py holds k = {config.KNN_K}, {config.KNN_WEIGHTS}  {status}")

    # Final model on ALL training candidates, using the config.py settings.
    features_train = training_table[config.KNN_FEATURES]
    scaler = clf.fit_feature_scaler(features_train)
    knn = clf.train_knn_classifier(scaler.transform(features_train), labels_train)
    clf.save_classifier(scaler, knn)
    print(f"    saved {config.SCALER_FILE.relative_to(config.PROJECT_ROOT)} and "
          f"{config.CLASSIFIER_FILE.relative_to(config.PROJECT_ROOT)}")

    # ---- 4. Classifier scores: out-of-fold (train) and test ----
    print("\n4) Classifier scores")
    print("    nested cross-validation: feature selection + k tuning repeated inside each outer fold "
          "(takes a few minutes)")
    model_for_case, out_of_fold, fold_log = train_nested_out_of_fold_models(training_table)
    save_table(fold_log, "step4_nested_selection.csv")
    for _, row in fold_log.iterrows():
        print(f"    fold {row['outer_fold']} (held out {row['held_out_cases']}): {row['n_features']} features, "
              f"k = {row['k']} {row['weights']}, inner CV F1 {row['inner_cv_f1']:.3f}")
    test_probability = clf.predict_kidney_probability(scaler, knn, test_table)
    score_rows = []
    for name, truth, probability in [("training, nested out-of-fold (CV)", labels_train, out_of_fold),
                                     ("TEST", test_table["label"].to_numpy(), test_probability)]:
        scores = metrics.binary_classification_scores(truth, (probability >= 0.5).astype(int))
        score_rows.append({"evaluated_on": name, "n_candidates": len(truth), **scores})
    score_table = pd.DataFrame(score_rows)
    save_table(score_table, "step4_classifier_test_results.csv")
    # Every candidate's P(kidney): nested out-of-fold for training, final model for test (used by step 5).
    predictions = pd.concat([training_table, test_table], ignore_index=True)[
        ["case_id", "slice_index", "split", "source", "seed_attempt", "label"]]
    predictions["p_kidney"] = np.concatenate([out_of_fold, test_probability])
    save_table(predictions, "step4_candidate_predictions.csv")
    print(score_table.round(3).to_string(index=False))

    # ---- Feature analysis (training only) ----
    print("\n5) Feature analysis (training candidates, all 17 features)")
    analysis = feature_analysis(training_table)
    save_table(analysis, "step4_feature_analysis.csv")
    print(analysis[["feature", "kidney_mean", "not_kidney_mean", "separation_abs_cohens_d"]].round(3).to_string(index=False))

    # ---- 5. Dice before / after the k-NN ----
    print("\n6) Dice before and after the k-NN (training: nested out-of-fold models; test: final model)")
    per_slice_train, kept_train = dice_before_after(train_slices, priors, lambda s: model_for_case[s["case_id"]])
    per_slice_test, kept_test = dice_before_after(test_slices, priors, lambda s: (scaler, knn))
    per_slice = pd.concat([per_slice_train, per_slice_test], ignore_index=True)
    kept = {**kept_train, **kept_test}
    save_table(per_slice, "step4_dice_per_slice.csv")
    dice_table = per_slice.groupby(["strategy", "method", "split"], sort=False).agg(
        dice_mean=("dice", "mean"), dice_median=("dice", "median"), dice_std=("dice", "std"),
        iou_mean=("iou", "mean"), slices_below_0_5=("dice", lambda d: int((d < 0.5).sum()))).reset_index()
    save_table(dice_table, "step4_dice_before_after_knn.csv")
    print(dice_table.round(3).to_string(index=False))

    # ---- Decision, on TRAINING Dice only (test numbers are never used to choose) ----
    train_dice = per_slice_train.groupby(["strategy", "method"])["dice"].mean()
    best_knn = max([("multi_otsu", "k-NN"), ("region_growing", "k-NN + seed retry")], key=lambda m: train_dice[m])
    best_rule = max([("multi_otsu", "rule filter"), ("region_growing", "rule filter")], key=lambda m: train_dice[m])
    use_knn = train_dice[best_knn] > train_dice[best_rule]
    chosen_strategy = best_knn[0] if use_knn else best_rule[0]
    print(f"\n    DECISION (training mean Dice): best k-NN pipeline {METHOD_NAMES[best_knn]} = {train_dice[best_knn]:.3f}; "
          f"best rule-filter pipeline {METHOD_NAMES[best_rule]} = {train_dice[best_rule]:.3f}")
    print(f"    -> USE_CLASSIFIER_IN_PIPELINE should be {use_knn}, strategy {chosen_strategy}")
    status = "OK" if (use_knn, chosen_strategy) == (config.USE_CLASSIFIER_IN_PIPELINE,
                                                    config.FINAL_SEGMENTATION_STRATEGY) else "<-- UPDATE config.py"
    print(f"    config.py: USE_CLASSIFIER_IN_PIPELINE = {config.USE_CLASSIFIER_IN_PIPELINE}, "
          f"FINAL_SEGMENTATION_STRATEGY = {config.FINAL_SEGMENTATION_STRATEGY}  {status}")

    # ---- Figures ----
    print("\n7) Figures")
    test_scores = score_rows[1]
    figure_confusion_matrix(test_scores)
    figure_feature_selection(selection_log)
    figure_nested_selection(fold_log, config.KNN_FEATURES)
    figure_cv_curve(cv_table, best)
    figure_feature_distributions(training_table, analysis)
    figure_candidate_decisions(test_slices, per_slice[per_slice["strategy"] == "multi_otsu"], kept)
    figure_seed_retry(train_slices + test_slices, per_slice, kept)


if __name__ == "__main__":
    main()
