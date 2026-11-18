"""
config.py - every tunable setting for the kidney pipeline, in one place.

WHY one file: anyone reading the code can see every number the pipeline uses
without hunting through the code. No other file should contain a "magic number";
it should import it from here instead.

RULE: every setting is tuned on the 8 TRAINING patients only. The 4 test
patients are touched once, at the very end (scripts/step5_evaluate_test_set.py).

Each setting has a comment saying what it controls and what happens if you
raise or lower it.
"""

from pathlib import Path

# =============================================================================
# 1. PATHS  (relative to this file, so the project runs on any computer)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"                # symlink -> data.nosync (kept out of iCloud)
RAW_DIR = DATA_DIR / "raw"                      # KiTS23 volumes: raw/case_000XX/imaging.nii.gz + segmentation.nii.gz
SLICES_DIR = DATA_DIR / "slices"                # the 72 extracted 2D slices (.npz, format in README.md)
SLICE_FILE_NAME = "case_{case:05d}_slice_{slice:03d}.npz"  # e.g. case_00012_slice_045.npz
SPLIT_FILE = DATA_DIR / "split.json"            # which patients are train / test

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"           # every figure (PNG)
TABLES_DIR = RESULTS_DIR / "tables"             # every CSV of numbers (PSNR, Dice, CV scores...)
MODELS_DIR = PROJECT_ROOT / "models"            # the trained scaler + k-NN, and learned seed offsets
DATASET_SUMMARY_CSV = TABLES_DIR / "step1_dataset_summary.csv"  # one row per slice (written by step1)

# Where KiTS23 is downloaded from (same sources the official kits23 repo uses).
KITS23_IMAGE_URL = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_{case:05d}.nii.gz"
KITS23_MASK_URL = "https://github.com/neheller/kits23/raw/main/dataset/case_{case:05d}/segmentation.nii.gz"

# =============================================================================
# 2. DATA CONVENTIONS
# =============================================================================
CASE_IDS = list(range(12, 24))                  # cases 12..23 = 12 patients

# Slices per patient, spread evenly from the top to the bottom of the kidney.
# 12 patients x 6 = 72 slices. Raising it gives more data but neighbouring
# slices become near-duplicates (less independent evidence).
SLICES_PER_PATIENT = 6

# Fraction of the kidney's top-to-bottom extent to skip at EACH end before
# spreading the slices. 0.1 = skip the outer 10% at the top and the bottom, where
# only a thin sliver of kidney shows. 0.0 would include those tips (very hard,
# unrepresentative cases); raising it concentrates the slices in the kidney's middle.
SLICE_EDGE_MARGIN_FRACTION = 0.1

# KiTS23 labels: 0 background, 1 kidney, 2 tumour, 3 cyst.
# Ground truth "whole kidney" = labels 1 + 2 + 3 (tumour and cyst sit inside the kidney).
KIDNEY_LABELS = (1, 2, 3)
TUMOUR_LABEL = 2
CYST_LABEL = 3

# Orientation every volume is turned into before slicing, as nibabel axis codes:
#   axis 0 runs towards I(nferior)  -> slice 0 is the top of the scan
#   axis 1 runs towards P(osterior) -> row 0 is the patient's front (front at the top)
#   axis 2 runs towards L(eft)      -> column 0 is the patient's right (standard radiology view)
# KiTS23 files are already stored this way (checked on case 12), so slice numbers
# match the original files.
DISPLAY_ORIENTATION = ("I", "P", "L")

# Train/test split BY PATIENT (never by slice - slices of one patient are too
# similar, mixing them would leak information and inflate results).
N_TRAIN_PATIENTS = 8                            # 8 train / 4 test = 67/33, closest to 70/30 with 12 patients
RANDOM_SEED = 42                                # fixed so the split, CV folds and figures are reproducible

# =============================================================================
# 3. PHASE 1 - ENHANCEMENT
# =============================================================================
# Every "chosen" value below was picked by scripts/step2_compare_enhancement.py on
# the 8 TRAINING patients. The *_CANDIDATES lists are what that script tries.

# --- HU window -> 0..255 ---
# Everything below LEVEL - WIDTH/2 becomes black, everything above LEVEL + WIDTH/2 white.
# Narrower WIDTH = more contrast between soft tissues but more clipping.
# Higher LEVEL = shifts the window towards brighter (denser) tissue.
# CHOSEN: [-50, 250] HU - highest CNR on 47 of 48 training slices (median 3.34 vs 2.90
# for the standard [-160, 240]). It wins because fat (about -100 HU) is clipped to pure
# black, so the kidney stands out from a flat background. Cost: about 7.5% of kidney
# pixels (brightest enhancing cortex, > 250 HU) are clipped to white.
HU_WINDOW_LEVEL = 100                           # centre of the window in HU
HU_WINDOW_WIDTH = 300                           # -> window = [-50, 250] HU
# Windows compared on training CNR, as (lowest HU, highest HU).
HU_WINDOW_CANDIDATES = [(-160, 240), (-100, 300), (-50, 250)]

# --- Perona-Malik anisotropic diffusion (edge-preserving smoothing) ---
# More iterations = smoother image, but edges slowly start to wash out too.
# CHOSEN: 80. With kappa 30 the edge ratio stays about 0.96 from 20 to 80 iterations
# while CNR keeps creeping up (+0.07 per doubling). Not pushed further: the gain is
# tiny and more diffusion flattens the kidney texture that Phase 3's GLCM measures.
ANISO_ITERATIONS = 80
# Kappa = edge threshold (in grey levels). Differences well above kappa are treated
# as edges and preserved; below it they are smoothed. Raise it = more smoothing
# (weak edges lost); lower it = less smoothing (noise kept).
# CHOSEN: 30 - best median CNR (3.67) that keeps >= 90% edge strength; kappa 50
# blurred edges below 0.9, kappa 10-20 barely smoothed.
ANISO_KAPPA = 30
# Step size per iteration. Must be <= 0.25 for the 4-neighbour scheme to be
# numerically stable. Larger = faster smoothing per iteration. Kept fixed: the
# amount of smoothing is controlled through ITERATIONS instead.
ANISO_GAMMA = 0.15
# Conduction function: 1 = exp(-(d/k)^2) (favours high-contrast edges),
# 2 = 1/(1+(d/k)^2) (favours wide regions over small ones). Both from Perona & Malik (1990).
# CHOSEN: 1. Function 2 lets more flow cross edges: its best CNR settings blurred edges to 0.7.
ANISO_CONDUCTION = 1
ANISO_KAPPA_CANDIDATES = [10, 20, 30, 50]
ANISO_ITERATION_CANDIDATES = [5, 10, 20, 40, 80]
ANISO_CONDUCTION_CANDIDATES = [1, 2]

# --- Median filter (the "standard" smoothing compared against) ---
# Odd kernel side length. Bigger = removes more noise but rounds off corners / thin structures.
# Best median kernel = 5 (median CNR 3.52, edge ratio 0.93), still below diffusion (3.67, 0.96).
# Kernel 7 blurs edges below 0.9. The median filter is only a comparison, not in the chain.
MEDIAN_KERNEL_SIZE = 5
MEDIAN_KERNEL_CANDIDATES = [3, 5, 7]

# Selection rule for smoothing: the highest CNR among settings that keep at least
# this fraction of the kidney's edge strength (metrics.edge_strength_on_boundary).
# Without it, CNR alone would always pick the blurriest setting.
# Raise it = only very gentle smoothing allowed; lower it = more blur tolerated.
MIN_EDGE_STRENGTH_RATIO = 0.9

# Which statistic over the 48 training slices decides every winner: "median" or "mean".
# Median, because a few slices whose background ring is almost all fat (clipped to
# black by a narrow window) get CNRs of 15-30 that would dominate a mean.
SELECTION_STATISTIC = "median"

# --- Homomorphic filter (frequency domain) ---
# H(u,v) = (gH - gL) * (1 - exp(-c * D^2 / D0^2)) + gL
# CHOSEN: gL 1.0, gH 1.25, D0 10 - the only mild setting family that raised CNR
# (median 3.94 vs 3.67 with no filter). Classic "illumination correction" (gL < 1)
# lowered CNR in 35 of 36 settings: CT values are calibrated HU, so there is no real
# illumination gradient to remove, and damping low frequencies shrinks the broad
# kidney-vs-fat contrast. Strong settings (gL 0.5, gH 2) cut CNR by about half.
# Gain for low frequencies (slow brightness changes). < 1 flattens gradients; 1 = untouched.
HOMOMORPHIC_GAMMA_LOW = 1.0
# Gain for high frequencies (detail/edges). > 1 sharpens; too high amplifies noise.
HOMOMORPHIC_GAMMA_HIGH = 1.25
# Cut-off radius D0 (in frequency pixels) between "low" and "high".
# Bigger D0 = more frequencies treated as illumination (stronger flattening).
HOMOMORPHIC_CUTOFF_D0 = 10
# Sharpness c of the transition. Note c and D0 do the same job: c * D^2 / D0^2 equals
# D^2 / (D0 / sqrt(c))^2, so c only rescales D0. We keep c = 1 and tune D0.
HOMOMORPHIC_SHARPNESS_C = 1.0
HOMOMORPHIC_GAMMA_LOW_CANDIDATES = [0.5, 0.75, 0.9, 1.0]
HOMOMORPHIC_GAMMA_HIGH_CANDIDATES = [1.0, 1.25, 1.5, 2.0]
HOMOMORPHIC_CUTOFF_D0_CANDIDATES = [10, 30, 80]
# After exp(), the image is stretched back to 0-255 between these percentiles
# (pixels beyond are clipped) so a few extreme pixels cannot squash the rest.
HOMOMORPHIC_OUTPUT_PERCENTILES = (0.5, 99.5)

# --- CLAHE vs global histogram equalisation ---
# Clip limit: how much each tile's histogram may peak before clipping.
# Higher = stronger contrast but more noise amplification; 1.0 = gentlest.
# CHOSEN: clip 1.0, 16x16 tiles - every CLAHE setting LOWERED CNR slightly (it also
# stretches the fat/muscle around the kidney); this one least (median 3.75 vs 3.94
# before CLAHE). Global histogram equalisation was far worse (2.39). CLAHE stays in the
# chain by design decision; revisit its settings with Dice in step 3.
CLAHE_CLIP_LIMIT = 1.0
# Number of tiles (rows, cols). More tiles = more local adaptation but more patchy look.
CLAHE_TILE_GRID = (16, 16)
CLAHE_CLIP_CANDIDATES = [1.0, 2.0, 3.0, 4.0]
CLAHE_TILE_CANDIDATES = [(4, 4), (8, 8), (16, 16)]

# --- CNR measurement ---
# Background for CNR = a ring of this many pixels just outside the ground-truth
# kidney (about 6-9 mm at these scans' pixel sizes). Wider ring = more background
# pixels (stabler std) but reaches into more unrelated organs.
CNR_BACKGROUND_RING_PX = 10

# =============================================================================
# 4. PHASE 2 - SEGMENTATION
# =============================================================================
# Every "CHOSEN" value below was picked by scripts/step3_compare_segmentation.py on
# the 8 TRAINING patients, using mean Dice (see SEGMENTATION_SELECTION_STATISTIC).

# --- Which image the segmentation sees (CLAHE vs HE vs none, compared by Dice) ---
# The Phase 1 chain up to the homomorphic filter, then a contrast step:
# "clahe", "histeq" (global histogram equalisation) or "none". Chosen per strategy
# by training Dice; the pipeline uses the one for FINAL_SEGMENTATION_STRATEGY.
# CHOSEN (training mean Dice, each strategy at its best settings):
#   multi-Otsu:      CLAHE 0.704 > none 0.699 > HE 0.590 (HE spreads every tissue over the
#                    full range, so the brightest class is no longer "kidney + vessels").
#   region growing:  HE 0.777 > CLAHE 0.632 > none 0.607. HE pushes the kidney into the top
#                    grey levels (230-255) so it becomes one near-uniform bright region that
#                    a seed can fill - even though HE had the WORST CNR in Phase 1.
CONTRAST_FOR_STRATEGY = {"multi_otsu": "clahe", "region_growing": "histeq"}
SEGMENTATION_CONTRAST_CANDIDATES = ["none", "clahe", "histeq"]

# Mean Dice (not median) picks Phase 2 winners: a setting that fails completely on
# some slices (Dice 0) must be penalised, and the mean does that. Dice is bounded
# 0..1, so it cannot explode on a few slices the way CNR did.
SEGMENTATION_SELECTION_STATISTIC = "mean"

# --- Body, bone and spine landmarks (found in RAW HU) ---
# Pixels above this HU are "inside the body" (air is about -1000 HU).
BODY_HU_THRESHOLD = -500
# Dense bone shell threshold. Training kidney: 0.8% of pixels > 300 HU, 0.01% > 400 HU,
# so 400 never takes kidney. Lower = risk of labelling enhancing kidney as bone;
# higher = thin bone shells are missed.
BONE_HU_THRESHOLD = 400
# Disk radius used to close small gaps in the bone shell before filling it.
BONE_CLOSING_RADIUS = 2
# Enclosed holes up to this size (pixels) are filled as bone: the spongy inside of a
# vertebra (150-300 HU, kidney-like once windowed). Larger holes are left alone so a
# ring of ribs + spine can never fill the whole abdomen.
BONE_MAX_HOLE_AREA_PX = 3000
# Bone is grown by this many pixels before being excluded, so its blurred border
# (partial volume) is not mistaken for bright kidney either.
BONE_EXCLUSION_MARGIN_PX = 2
# Half-width of the band (fraction of body width) around the body's centre in which
# the spine is searched. Wider = finds a spine bent off-centre, but risks a rib.
SPINE_SEARCH_HALF_WIDTH_FRACTION = 0.12

# --- Strategy A: multi-Otsu thresholding ---
# Number of intensity classes (3 or more). More classes = finer split of
# tissue types, but each class becomes noisier and less stable.
# CHOSEN: 3 (Dice 0.704 vs 0.674 for 4 and 0.641 for 5, CLAHE input). With more classes
# the bright kidney cortex and the darker inner kidney are split into separate classes.
MULTI_OTSU_CLASSES = 3
# How many of the BRIGHTEST classes count as kidney candidates (1 = only the top class).
# CHOSEN: 1. Taking 2 of 3 classes takes almost all soft tissue (liver, muscle, bowel).
MULTI_OTSU_KIDNEY_CLASSES = 1
MULTI_OTSU_CLASS_CANDIDATES = [3, 4, 5]
MULTI_OTSU_KIDNEY_CLASS_CANDIDATES = [1, 2]

# --- Strategy B: seeded region growing ---
# A neighbour joins the region if |its grey level - region mean| < tolerance.
# Bigger = region grows further (risk of leaking into liver/spleen/bowel);
# smaller = region stops early (under-segmentation).
# CHOSEN: 20 on the HE input (Dice 0.777; 30 -> 0.750, 10 -> 0.693, 50 -> 0.646 as leaks grow).
REGION_GROW_TOLERANCE = 20
REGION_GROW_TOLERANCE_CANDIDATES = [10, 20, 30, 40, 50]
# Region size cap = largest training kidney x this factor. Reaching it means a leak.
REGION_GROW_MAX_AREA_FACTOR = 1.5
# Seed check: a seed is kidney-like if its grey level lies between these percentiles
# of the training kidneys' grey levels. Narrower = stricter (more seeds moved/skipped).
# CHOSEN: (40, 99). A high low-end pushes seeds onto the bright outer cortex instead of
# the darker inner kidney; the cortex ring is grown and fill_holes completes the kidney.
# Low end 10 -> Dice 0.548, 40 -> 0.777, 60 -> 0.763 (tolerance 20, HE input).
KIDNEY_INTENSITY_PERCENTILES = (40, 99)
KIDNEY_SEED_LOW_PERCENTILE_CANDIDATES = [10, 40, 60]
# If a seed is not kidney-like, search this far around it (fraction of body width,
# about 20-30 pixels) for one that is. Bigger = finds displaced kidneys but may
# grab a neighbouring organ on slices with only one kidney.
SEED_SEARCH_RADIUS_FRACTION = 0.07

# --- Rule filter (keeps one blob per side; replaced by the k-NN in Phase 3) ---
# Learned position/size zones are widened by this fraction of their range on each end.
# Bigger = accepts more unusual kidneys, but also more look-alike blobs.
ZONE_MARGIN_FRACTION = 0.10
SEGMENTATION_PRIORS_FILE = MODELS_DIR / "segmentation_priors.json"  # learned seeds, zones, intensity range

# --- My own structuring elements (explicit numpy arrays built in phase2) ---
# Radius (pixels) of the disk used for OPENING: cuts thin bridges that join kidney
# to bowel, liver or spleen. Bigger = cuts more bridges but erodes thin kidney parts.
# CHOSEN per strategy: multi-Otsu 2 (its blobs are already separate; a bigger disk
# erodes the thin cortex rim), region growing 3 (cuts leaks along renal vessels;
# 5 starts eating small kidneys: 0.741).
OPENING_RADIUS_FOR_STRATEGY = {"multi_otsu": 2, "region_growing": 3}
SE_OPENING_RADIUS_CANDIDATES = [2, 3, 5]
# Radius of the disk used for CLOSING: fills small gaps along the kidney border.
# Bigger = smoother border but may merge kidney with a touching organ.
SE_CLOSING_RADIUS = 5
# Size (rows, cols) of the vertical ellipse that matches the kidney's elongated
# shape in axial slices. Used for the final smoothing opening.
SE_KIDNEY_ELLIPSE_SIZE = (9, 5)

# Objects smaller than this (pixels) are removed as noise. The smallest training
# kidney piece kept as ground truth is also this size.
# Bigger = cleaner masks but very small kidney tips (top/bottom slices) are lost.
MIN_OBJECT_AREA_PX = 100

# Which strategy the final pipeline (src/pipeline.py) uses: "multi_otsu" or "region_growing".
# CHOSEN: region growing (training mean Dice 0.777 vs 0.704 for multi-Otsu).
FINAL_SEGMENTATION_STRATEGY = "region_growing"
# The contrast step and opening radius the pipeline uses = the ones chosen for that strategy.
SEGMENTATION_CONTRAST = CONTRAST_FOR_STRATEGY[FINAL_SEGMENTATION_STRATEGY]
SE_OPENING_RADIUS = OPENING_RADIUS_FOR_STRATEGY[FINAL_SEGMENTATION_STRATEGY]

# =============================================================================
# 5. PHASE 3 - FEATURES AND k-NN CLASSIFIER
# =============================================================================
# A candidate blob is labelled "kidney" (1) if at least this fraction of its
# pixels lie inside the ground-truth kidney; otherwise "not kidney" (0).
# Higher = stricter positive class.
BLOB_KIDNEY_OVERLAP_FRACTION = 0.5
# Blobs smaller than this are not even considered as candidates.
MIN_CANDIDATE_AREA_PX = MIN_OBJECT_AREA_PX

# GLCM: computed on the WINDOWED image (before diffusion, which flattens texture).
# Number of grey levels the blob is quantised to before counting pairs.
# Fewer levels = more pairs per cell (stabler statistics on small blobs) but less detail.
# 32 levels over the [-50, 250] HU window = about 9 HU per level.
GLCM_LEVELS = 32
# Pixel distance between the two pixels in each pair. 1 = immediate neighbours (fine texture).
GLCM_DISTANCES = [1]
# The four standard directions.
GLCM_ANGLES_DEGREES = [0, 45, 90, 135]

# Every "CHOSEN" value below was picked by scripts/step4_train_classifier.py on the
# 8 TRAINING patients, by 5-fold cross-validation grouped by patient.

# Number of neighbours that vote. Small k = follows the training data closely (can
# memorise noise); large k = smoother decision but small groups get outvoted.
# Odd numbers avoid ties in a 2-class vote.
# CHOSEN: 7 (CV F1 0.761 ± 0.116). Smaller k memorises single look-alike candidates;
# larger k lets the far more numerous not-kidney candidates outvote real kidneys.
KNN_K = 7
KNN_K_CANDIDATES = list(range(1, 26, 2))        # 1, 3, 5, ..., 25
# "uniform" = each neighbour gets one vote; "distance" = closer neighbours count more (1/distance).
# CHOSEN: uniform (best F1 at k = 7).
KNN_WEIGHTS = "uniform"
KNN_WEIGHT_CANDIDATES = ["uniform", "distance"]
# Distance used to find neighbours. Euclidean is the natural choice after standardising.
KNN_METRIC = "euclidean"
# Number of cross-validation folds (5). Folds hold out whole patients (GroupKFold).
CV_FOLDS = 5
# Which score picks k and the weighting. F1 copes with class imbalance (far more
# not-kidney candidates than kidney ones); accuracy would reward saying "not kidney".
CV_SCORING = "f1"

# k-NN setting used WHILE selecting features: the k chosen by cross-validation on all 17
# features (first run of step 4). Fixed here so selection and k-tuning do not chase each other.
FEATURE_SELECTION_K = 7
FEATURE_SELECTION_WEIGHTS = "uniform"
# Features dropped BEFORE backward selection: compactness = 4*pi / circularity, so it
# repeats circularity and the k-NN distance would count that shape property twice.
KNN_FEATURES_DROPPED_BEFORE_SELECTION = ["compactness"]
# The features the k-NN actually uses (all 17 are still computed and analysed).
# CHOSEN by backward selection on grouped 5-fold CV F1 (training only): starting from
# 16 features (CV F1 0.748), removing 9 one at a time raised it to 0.878. Removed, in
# order: glcm_energy, hu_mean, glcm_entropy, glcm_contrast, solidity, glcm_homogeneity,
# glcm_correlation, eccentricity, perimeter_to_area. Several are near-copies of a kept
# feature (perimeter/area of area + circularity; the GLCM measures of each other and of
# HU variance), and every extra feature adds noise to the k-NN distance.
# Caution: chosen on the same 8 patients as the CV score, so 0.878 is optimistic. The
# honest, nested estimate (selection repeated inside each fold) is CV F1 0.763. The
# selection is unstable: folds kept 6-13 features; only dx, dy, hu_variance and
# perimeter_to_area were chosen in all 5 folds (step4_nested_selection.csv).
KNN_FEATURES = ["area", "circularity", "hu_variance", "hu_skewness", "hu_kurtosis",
                "dx_from_spine", "dy_from_spine"]

# Strategy B with the k-NN: how many seeds to try per side. If the k-NN rejects
# the region grown from seed 1, the nearest kidney-like pixel outside that region
# becomes seed 2, and so on. More = more chances to find a displaced kidney, but
# also more chances to accept a look-alike on a side that has no kidney.
SEED_MAX_ATTEMPTS = 3

# Saved model files.
CLASSIFIER_FILE = MODELS_DIR / "knn_classifier.joblib"
SCALER_FILE = MODELS_DIR / "feature_scaler.joblib"

# Whether pipeline.segment_kidney() lets the k-NN pick the kidney candidates
# (True) or uses the Phase 2 rule filter (False). Needs models/ from step 4.
# Either way the k-NN reports P(kidney) for each final region (segment_kidney_with_details),
# so the classifier stays in the pipeline as a check that never deletes a region.
# CHOSEN on TRAINING Dice only, with NESTED cross-validation (feature selection and k
# tuning repeated inside each fold): region growing + k-NN + seed retry 0.745 vs
# region growing + rule filter 0.777 -> False. (Without nesting the k-NN looked better,
# 0.819, because feature selection had seen the held-out patients.) Test not used.
USE_CLASSIFIER_IN_PIPELINE = False

# =============================================================================
# 6. FIGURES
# =============================================================================
FIGURE_DPI = 150                                # resolution of saved PNGs
PREDICTED_CONTOUR_COLOUR = "red"                # detected mask = red contour
GROUND_TRUTH_CONTOUR_COLOUR = "lime"            # ground truth = green contour
N_BEST_EXAMPLES = 3                             # how many best-Dice slices to save as figures
N_WORST_EXAMPLES = 6                            # how many worst-Dice TEST slices (failure cases) to save per strategy
EXAMPLE_TRAIN_SLICE = (23, 57)                  # (case, slice) used for training example figures (a mid-kidney slice)
EXAMPLE_HARD_TEST_SLICE = (22, 239)             # (case, slice) from the hardest test patient (noisy 0.5 mm scan)

# =============================================================================
# 7. FINAL EVALUATION (scripts/step5_evaluate_test_set.py) - analysis only, no tuning
# =============================================================================
# Test slices below this Dice are analysed as failures in results/failure_analysis.md.
FAILURE_DICE_THRESHOLD = 0.85
# A ground-truth kidney piece smaller than this (pixels) counts as a "sliver" (top or
# bottom tip of the kidney). 1000 px is below the 7th percentile of single-kidney areas.
FAILURE_SLIVER_AREA_PX = 1000
# Where the 24 test slices are exported, so results can be checked without the full dataset.
SAMPLE_TEST_IMAGES_DIR = PROJECT_ROOT / "sample_test_images"
FAILURE_ANALYSIS_FILE = RESULTS_DIR / "failure_analysis.md"
