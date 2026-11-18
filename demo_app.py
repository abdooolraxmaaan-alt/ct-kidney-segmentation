"""
demo_app.py - a small desktop app to try the kidney pipeline on CT slices.

It calls the same function other programs use (src.pipeline.segment_kidney_with_details),
so what you see here is exactly what the pipeline produces.

Run from the project folder:
    .venv/bin/python demo_app.py                          (then use the buttons)
    .venv/bin/python demo_app.py sample_test_images       (open a folder straight away)
    .venv/bin/python demo_app.py path/to/slice.npz        (open one slice)
(Windows: .venv\\Scripts\\python demo_app.py)

Shows:
    - original slice (soft-tissue window), enhanced slice (Phase 1 output), predicted
      mask, and overlay (red = predicted, green = ground truth)
    - Dice and IoU for the slice, P(kidney) for each region found
    - the classifier's test metrics (accuracy, sensitivity, specificity, F1)

Needs Tkinter (part of the standard Python installer on Windows and python.org macOS;
Homebrew Python: brew install python-tk@3.14; Ubuntu: sudo apt install python3-tk).
"""

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib

matplotlib.use("TkAgg")  # draw matplotlib figures inside the Tkinter window
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import config  # noqa: E402
from src import metrics, phase1_enhancement  # noqa: E402
from src.pipeline import plot_contour_overlay, segment_kidney_with_details  # noqa: E402


def default_folder() -> Path:
    """Folder the file dialogs start in: the exported test slices, else the prepared slices."""
    for folder in (config.SAMPLE_TEST_IMAGES_DIR, config.SLICES_DIR, config.PROJECT_ROOT):
        if folder.exists():
            return folder
    return Path.cwd()


def read_slice_file(path: Path) -> dict:
    """Read one .npz slice. The ground truth is optional, so slices without it can be segmented too.

    Outputs:
        {"hu": float32 HU array, "ground_truth": bool mask or None, "case_id", "slice_index"}
    """
    with np.load(path) as data:
        return {
            "hu": data["hu"].astype(np.float32),
            "ground_truth": data["kidney_mask"].astype(bool) if "kidney_mask" in data.files else None,
            "case_id": int(data["case_id"]) if "case_id" in data.files else None,
            "slice_index": int(data["slice_index"]) if "slice_index" in data.files else None,
        }


def load_classifier_test_metrics() -> dict | None:
    """The k-NN's scores on the test candidates, read from results/tables (None if not computed yet)."""
    table_path = config.TABLES_DIR / "step4_classifier_test_results.csv"
    if not table_path.exists():
        return None
    table = pd.read_csv(table_path)
    test_row = table[table["evaluated_on"] == "TEST"].iloc[0]
    return {"n_candidates": int(test_row["n_candidates"]), "accuracy": test_row["accuracy"],
            "sensitivity": test_row["recall"], "specificity": test_row["specificity"], "f1": test_row["f1"]}


class KidneyDemoApp:
    """The main window: buttons on top, four image views on the left, results on the right."""

    def __init__(self, root: tk.Tk):
        """Build the window and remember which slices are open."""
        self.root = root
        self.root.title("Kidney segmentation - demo (KiTS23)")
        self.slice_paths: list[Path] = []
        self.current_index = 0
        self.classifier_metrics = load_classifier_test_metrics()
        self.build_controls()
        self.build_views()
        self.build_info_panel()
        self.show_message("Open a slice (.npz) or a folder of slices to start.")

    # ---- window layout -------------------------------------------------------
    def build_controls(self) -> None:
        """Top bar: open buttons, previous/next, and the name of the current file."""
        bar = tk.Frame(self.root, padx=8, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="Open slice", command=self.open_slice).pack(side=tk.LEFT)
        tk.Button(bar, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=4)
        self.previous_button = tk.Button(bar, text="< Previous", command=self.show_previous, state=tk.DISABLED)
        self.previous_button.pack(side=tk.LEFT, padx=(16, 4))
        self.next_button = tk.Button(bar, text="Next >", command=self.show_next, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT)
        self.file_label = tk.Label(bar, text="", anchor="w")
        self.file_label.pack(side=tk.LEFT, padx=16)

    def build_views(self) -> None:
        """Left side: a 2x2 matplotlib figure for the four views."""
        self.figure = Figure(figsize=(9, 8.6), dpi=100)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def build_info_panel(self) -> None:
        """Right side: a read-only text panel for the numbers."""
        self.info_text = tk.Text(self.root, width=44, font=("Courier", 13), padx=10, pady=10, relief=tk.FLAT)
        self.info_text.pack(side=tk.RIGHT, fill=tk.Y)

    def show_message(self, message: str) -> None:
        """Replace the text panel's contents."""
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.configure(state=tk.DISABLED)

    # ---- opening files -------------------------------------------------------
    def open_slice(self) -> None:
        """Ask for one .npz file and show it."""
        path = filedialog.askopenfilename(title="Open a slice", initialdir=default_folder(),
                                          filetypes=[("CT slice", "*.npz")])
        if path:
            self.show_files([Path(path)])

    def open_folder(self) -> None:
        """Ask for a folder and show the first .npz slice in it (Previous/Next move through the rest)."""
        folder = filedialog.askdirectory(title="Open a folder of slices", initialdir=default_folder())
        if folder:
            self.open_path(Path(folder))

    def open_path(self, path: Path) -> None:
        """Open a file or a folder given on the command line or by a dialog."""
        paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        if not paths:
            messagebox.showwarning("No slices", f"No .npz slices found in {path}")
            return
        self.show_files(paths)

    def show_files(self, paths: list[Path], index: int = 0) -> None:
        """Remember the list of slices and show one of them."""
        self.slice_paths = paths
        self.current_index = index
        self.show_current_slice()

    def show_previous(self) -> None:
        """Go one slice back in the open folder."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_slice()

    def show_next(self) -> None:
        """Go one slice forward in the open folder."""
        if self.current_index < len(self.slice_paths) - 1:
            self.current_index += 1
            self.show_current_slice()

    # ---- running the pipeline and showing the result ------------------------
    def show_current_slice(self) -> None:
        """Run the pipeline on the current slice and update every view and number."""
        path = self.slice_paths[self.current_index]
        self.file_label.configure(text=f"{path.name}   ({self.current_index + 1} of {len(self.slice_paths)})")
        self.previous_button.configure(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.configure(
            state=tk.NORMAL if self.current_index < len(self.slice_paths) - 1 else tk.DISABLED)
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            slice_data = read_slice_file(path)
            start = time.perf_counter()
            details = segment_kidney_with_details(slice_data["hu"])
            runtime = time.perf_counter() - start
            enhanced = phase1_enhancement.enhance_slice(slice_data["hu"])
        except Exception as error:  # show the problem instead of closing the app
            messagebox.showerror("Could not process the slice", str(error))
            return
        finally:
            self.root.configure(cursor="")
        self.draw_views(slice_data, enhanced, details)
        self.show_message(self.results_text(path, slice_data, details, runtime))

    def draw_views(self, slice_data: dict, enhanced: np.ndarray, details: dict) -> None:
        """Original, enhanced, predicted mask and overlay."""
        windowed = phase1_enhancement.apply_hu_window(slice_data["hu"])
        low = config.HU_WINDOW_LEVEL - config.HU_WINDOW_WIDTH / 2
        for axis in self.axes.ravel():
            axis.clear()
            axis.axis("off")

        self.axes[0, 0].imshow(windowed, cmap="gray")
        self.axes[0, 0].set_title(f"Original slice (window [{low:g}, {low + config.HU_WINDOW_WIDTH:g}] HU)", fontsize=10)
        self.axes[0, 1].imshow(enhanced, cmap="gray")
        self.axes[0, 1].set_title(f"Enhanced (Phase 1, {config.SEGMENTATION_CONTRAST})", fontsize=10)

        self.axes[1, 0].imshow(details["mask"], cmap="gray")
        for region in details["regions"]:
            rows, cols = np.nonzero(region["mask"])
            self.axes[1, 0].text(cols.mean(), rows.min() - 10, f"P={region['p_kidney']:.2f}", color="cyan",
                                 ha="center", fontsize=9, fontweight="bold")
        self.axes[1, 0].set_title("Predicted mask (P = k-NN P(kidney))", fontsize=10)

        plot_contour_overlay(self.axes[1, 1], windowed, details["mask"], slice_data["ground_truth"],
                             "Overlay: red = predicted, green = ground truth")
        self.figure.tight_layout()
        self.canvas.draw()

    def results_text(self, path: Path, slice_data: dict, details: dict, runtime: float) -> str:
        """The numbers for the right-hand panel."""
        lines = [f"Case {slice_data['case_id']}, slice {slice_data['slice_index']}", ""]
        truth = slice_data["ground_truth"]
        if truth is not None:
            lines += ["Segmentation (this slice)",
                      f"  Dice   {metrics.dice_coefficient(details['mask'], truth):.3f}",
                      f"  IoU    {metrics.iou_score(details['mask'], truth):.3f}",
                      f"  Area   predicted {int(details['mask'].sum())} px",
                      f"         truth     {int(truth.sum())} px", ""]
        else:
            lines += ["(no ground truth in this file:", " Dice and IoU not available)", ""]

        lines.append("Regions found (k-NN check)")
        if not details["regions"]:
            lines.append("  none")
        for region in details["regions"]:
            verdict = "kidney" if region["p_kidney"] >= 0.5 else "NOT kidney?"
            lines.append(f"  {region['side']:5s} {region['area_px']:5d} px  P={region['p_kidney']:.2f}  {verdict}")
        method = "k-NN" if config.USE_CLASSIFIER_IN_PIPELINE else "rule filter"
        lines += ["", f"Pipeline: {config.FINAL_SEGMENTATION_STRATEGY.replace('_', ' ')}",
                  f"          + {method}, k-NN check", f"Time:     {runtime:.2f} s", ""]

        if self.classifier_metrics:
            m = self.classifier_metrics
            lines += [f"Classifier on {m['n_candidates']} test candidates",
                      f"  Accuracy     {m['accuracy']:.3f}",
                      f"  Sensitivity  {m['sensitivity']:.3f} (recall)",
                      f"  Specificity  {m['specificity']:.3f}",
                      f"  F1           {m['f1']:.3f}"]
        else:
            lines.append("(classifier metrics not found:\n run scripts/step4_train_classifier.py)")
        return "\n".join(lines)


def save_window_screenshot(root: tk.Tk, output_path: Path) -> None:
    """Save a picture of the app window (used to make the README screenshot).

    Uses Pillow's ImageGrab (Pillow comes with matplotlib). On macOS the terminal
    needs Screen Recording permission, otherwise the picture may come out blank.
    """
    from PIL import ImageGrab

    root.update()
    left, top = root.winfo_rootx(), root.winfo_rooty()
    box = (left, top, left + root.winfo_width(), top + root.winfo_height())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ImageGrab.grab(bbox=box).save(output_path)
    print(f"saved {output_path}")


def main() -> None:
    """Start the app; optionally open a slice/folder and save a screenshot, then close."""
    parser = argparse.ArgumentParser(description="Kidney segmentation demo app")
    parser.add_argument("path", nargs="?", help="a .npz slice or a folder of slices to open")
    parser.add_argument("--index", type=int, default=0, help="which slice of the folder to show first (0 = first)")
    parser.add_argument("--screenshot", help="save a screenshot of the window to this file, then close")
    arguments = parser.parse_args()

    root = tk.Tk()
    app = KidneyDemoApp(root)
    if arguments.path:
        path = Path(arguments.path)
        paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        app.show_files(paths, min(arguments.index, len(paths) - 1))
    if arguments.screenshot:
        # Wait for the window to be drawn on screen, then capture it and close.
        root.after(2500, lambda: (save_window_screenshot(root, Path(arguments.screenshot)), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
