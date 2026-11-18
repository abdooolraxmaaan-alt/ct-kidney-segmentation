"""
phase1_enhancement.py - Phase 1: CT preprocessing and spatial / frequency enhancement.

Phase 1 covers:
    1. HU window, normalised to 8-bit 0-255             -> apply_hu_window
    2. edge-preserving filter vs standard smoothing      -> anisotropic_diffusion_perona_malik vs median_filter
       CLAHE vs histogram equalisation                   -> clahe_enhancement vs histogram_equalisation
    3. 2D DFT log-magnitude spectrum                     -> dft_log_magnitude_spectrum
       homomorphic filter                                -> gaussian_high_emphasis_filter, homomorphic_filter
    4. PSNR and CNR                                      -> see metrics.py

The chain actually used by the pipeline is enhance_slice().
Every setting comes from config.py and was chosen on TRAINING patients only
(scripts/step2_compare_enhancement.py).

Textbook references ("Textbook: Ch. N") are to Gonzalez, R. C., & Woods, R. E. (2018).
Digital image processing (4th ed.). Pearson.
"""

import cv2
import numpy as np

import config


# -----------------------------------------------------------------------------
# 1. HU window
# -----------------------------------------------------------------------------
def apply_hu_window(hu_slice: np.ndarray,
                    level: float = config.HU_WINDOW_LEVEL,
                    width: float = config.HU_WINDOW_WIDTH) -> np.ndarray:
    """Clip raw Hounsfield Units to a soft-tissue window and stretch it to 0-255.

    Inputs:
        hu_slice: 2D array of raw HU values.
        level, width: window centre and width in HU (see config).
    Outputs:
        2D uint8 image: HU <= level - width/2 -> 0, HU >= level + width/2 -> 255,
        linear in between.
    Textbook: Ch. 3 (Intensity Transformations and Spatial Filtering) - contrast stretching / intensity windowing.
    """
    # WHY a window: CT spans about -1000 (air) to +3000 HU (dense bone/metal). Mapping
    # all of that to 0-255 would squeeze every soft tissue (kidney, liver, muscle,
    # roughly -100..250 HU) into a handful of grey levels.
    lowest_hu = level - width / 2
    highest_hu = level + width / 2
    clipped = np.clip(hu_slice, lowest_hu, highest_hu)

    # Linear stretch: lowest_hu -> 0, highest_hu -> 255.
    scaled = (clipped - lowest_hu) / (highest_hu - lowest_hu) * 255.0
    return np.round(scaled).astype(np.uint8)


# -----------------------------------------------------------------------------
# 2a. Noise removal: anisotropic diffusion vs median
# -----------------------------------------------------------------------------
def perona_malik_conduction(gradient: np.ndarray, kappa: float, conduction: int) -> np.ndarray:
    """How much "heat" may flow across a given intensity difference.

    Near 1 for small differences (noise -> smooth it), near 0 for large
    differences (an organ edge -> leave it alone). Kappa sets where the switch happens.

        conduction 1: g = exp(-(gradient / kappa)^2)      favours high-contrast edges
        conduction 2: g = 1 / (1 + (gradient / kappa)^2)  favours wide regions

    Inputs:
        gradient: array of differences between a pixel and one neighbour.
        kappa: edge threshold in grey levels.
        conduction: 1 or 2 (which formula).
    Outputs:
        Array of conduction coefficients in (0, 1].
    Reference:
        Perona & Malik (1990), equations for g(.) - see anisotropic_diffusion_perona_malik.
    """
    ratio_squared = (gradient / kappa) ** 2
    if conduction == 1:
        return np.exp(-ratio_squared)
    if conduction == 2:
        return 1.0 / (1.0 + ratio_squared)
    raise ValueError("conduction must be 1 or 2")


def anisotropic_diffusion_perona_malik(image: np.ndarray,
                                       iterations: int = config.ANISO_ITERATIONS,
                                       kappa: float = config.ANISO_KAPPA,
                                       gamma: float = config.ANISO_GAMMA,
                                       conduction: int = config.ANISO_CONDUCTION) -> np.ndarray:
    """Edge-preserving smoothing: blur inside regions but not across edges.

    Each iteration, every pixel looks at its 4 neighbours (north, south, east, west).
    It moves towards each neighbour by  gamma * g(difference) * difference,
    where g() is near 1 for small differences (noise) and near 0 for large ones
    (edges). So noise is averaged away while the kidney/fat boundary stays sharp.

        I_new = I + gamma * (gN*dN + gS*dS + gE*dE + gW*dW)

    Written by hand with numpy (no library call) so every step can be explained.

    Inputs:
        image: 2D uint8 windowed slice.
        iterations: number of update steps (more = smoother).
        kappa: edge threshold in grey levels (see perona_malik_conduction).
        gamma: step size, must be <= 0.25 for the 4-neighbour scheme to be stable.
        conduction: 1 or 2 (see perona_malik_conduction).
    Outputs:
        2D uint8 smoothed image.
    Reference:
        Perona, P., & Malik, J. (1990). Scale-space and edge detection using
        anisotropic diffusion. IEEE Transactions on Pattern Analysis and Machine
        Intelligence, 12(7), 629-639.
    """
    if gamma > 0.25:
        # With 4 neighbours, a larger step can overshoot and make the image oscillate.
        raise ValueError("gamma must be <= 0.25 for a stable 4-neighbour update")

    # Work in float so small fractional updates are not lost to rounding each iteration.
    smoothed = image.astype(np.float64)

    for _ in range(iterations):
        # Difference from each pixel to its neighbour in each direction.
        # Pixels on the image border have no neighbour outside, so their difference
        # in that direction stays 0 (nothing flows in or out of the image).
        difference_north = np.zeros_like(smoothed)
        difference_north[1:, :] = smoothed[:-1, :] - smoothed[1:, :]

        difference_south = np.zeros_like(smoothed)
        difference_south[:-1, :] = smoothed[1:, :] - smoothed[:-1, :]

        difference_east = np.zeros_like(smoothed)
        difference_east[:, :-1] = smoothed[:, 1:] - smoothed[:, :-1]

        difference_west = np.zeros_like(smoothed)
        difference_west[:, 1:] = smoothed[:, :-1] - smoothed[:, 1:]

        # Flow from each direction = conduction coefficient x difference.
        total_flow = (
            perona_malik_conduction(difference_north, kappa, conduction) * difference_north
            + perona_malik_conduction(difference_south, kappa, conduction) * difference_south
            + perona_malik_conduction(difference_east, kappa, conduction) * difference_east
            + perona_malik_conduction(difference_west, kappa, conduction) * difference_west
        )
        smoothed = smoothed + gamma * total_flow

    return np.clip(np.round(smoothed), 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = config.MEDIAN_KERNEL_SIZE) -> np.ndarray:
    """Standard non-linear smoothing: replace each pixel by the median of its neighbourhood.

    The standard baseline that anisotropic diffusion is compared against.

    Inputs:
        image: 2D uint8 windowed slice.
        kernel_size: odd window side length (see config).
    Outputs:
        2D uint8 smoothed image.
    Textbook: Ch. 3 (Intensity Transformations and Spatial Filtering) - order-statistic / median filter.
    """
    return cv2.medianBlur(image, kernel_size)


# -----------------------------------------------------------------------------
# 2b. Contrast: global histogram equalisation vs CLAHE
# -----------------------------------------------------------------------------
def histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """Global histogram equalisation: remap grey levels so the histogram is roughly flat.

    Every grey level v is sent to 255 * CDF(v), so crowded grey levels get spread apart.
    One mapping is used for the whole image.

    Inputs:
        image: 2D uint8 image.
    Outputs:
        2D uint8 equalised image.
    Textbook: Ch. 3 (Intensity Transformations and Spatial Filtering) - histogram equalisation.
    """
    return cv2.equalizeHist(image)


def clahe_enhancement(image: np.ndarray,
                      clip_limit: float = config.CLAHE_CLIP_LIMIT,
                      tile_grid: tuple[int, int] = config.CLAHE_TILE_GRID) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the windowed soft-tissue slice.

    Equalises each tile separately (local contrast), but clips each tile's histogram
    at clip_limit first so that noise in flat areas is not blown up. Tiles are blended
    with bilinear interpolation so no tile borders show.

    Inputs:
        image: 2D uint8 windowed slice.
        clip_limit, tile_grid: see config.
    Outputs:
        2D uint8 enhanced image.
    Textbook: Ch. 3 (Intensity Transformations and Spatial Filtering) - local histogram processing.
        CLAHE: Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization.
        In P. S. Heckbert (Ed.), Graphics gems IV (pp. 474-485). Academic Press.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    return clahe.apply(image)


# -----------------------------------------------------------------------------
# 3. Frequency domain: DFT spectrum and homomorphic filter
# -----------------------------------------------------------------------------
def dft_log_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute the 2D Discrete Fourier Transform and return its centred log-magnitude spectrum.

    Inputs:
        image: 2D slice (uint8 or float).
    Outputs:
        2D float array log(1 + |F(u, v)|), shifted so zero frequency is in the centre.
        The log is needed for display: the centre value is millions of times larger
        than the rest and would hide everything else.
    Textbook: Ch. 4 (Filtering in the Frequency Domain) - DFT, fftshift, log transform for display.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    return np.log1p(np.abs(spectrum))


def gaussian_high_emphasis_filter(shape: tuple[int, int],
                                  gamma_low: float = config.HOMOMORPHIC_GAMMA_LOW,
                                  gamma_high: float = config.HOMOMORPHIC_GAMMA_HIGH,
                                  cutoff_d0: float = config.HOMOMORPHIC_CUTOFF_D0,
                                  sharpness_c: float = config.HOMOMORPHIC_SHARPNESS_C) -> np.ndarray:
    """Build the homomorphic filter H(u, v) for a centred spectrum.

        H(u, v) = (gamma_high - gamma_low) * (1 - exp(-c * D^2 / D0^2)) + gamma_low

    D(u, v) is the distance from the centre of the spectrum. At D = 0 (slow
    brightness changes) H = gamma_low; far from the centre (fine detail) H -> gamma_high.

    Inputs:
        shape: (rows, cols) of the image.
        gamma_low, gamma_high, cutoff_d0, sharpness_c: see config (HOMOMORPHIC_*).
    Outputs:
        2D float array H, same shape as the image, centred like fftshift output.
    Textbook: Ch. 4 (Filtering in the Frequency Domain) - Gaussian filter transfer functions.
    """
    rows, cols = shape
    # Frequency coordinates measured from the centre of the shifted spectrum.
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    v_grid, u_grid = np.meshgrid(v, u)
    distance_squared = u_grid ** 2 + v_grid ** 2

    return (gamma_high - gamma_low) * (1 - np.exp(-sharpness_c * distance_squared / cutoff_d0 ** 2)) + gamma_low


def rescale_to_uint8(image: np.ndarray,
                     percentiles: tuple[float, float] = config.HOMOMORPHIC_OUTPUT_PERCENTILES) -> np.ndarray:
    """Stretch a float image back to 0-255, ignoring a few extreme pixels.

    The low/high percentiles map to 0/255 and anything beyond is clipped. Using
    percentiles instead of the exact min/max stops a handful of overshoot pixels
    (e.g. at the body/air border) from squashing every other grey level.

    Inputs:
        image: 2D float image.
        percentiles: (low, high) in percent.
    Outputs:
        2D uint8 image.
    """
    low_value, high_value = np.percentile(image, percentiles)
    if high_value <= low_value:
        return np.zeros(image.shape, dtype=np.uint8)
    stretched = (np.clip(image, low_value, high_value) - low_value) / (high_value - low_value) * 255.0
    return np.round(stretched).astype(np.uint8)


def homomorphic_filter(image: np.ndarray,
                       gamma_low: float = config.HOMOMORPHIC_GAMMA_LOW,
                       gamma_high: float = config.HOMOMORPHIC_GAMMA_HIGH,
                       cutoff_d0: float = config.HOMOMORPHIC_CUTOFF_D0,
                       sharpness_c: float = config.HOMOMORPHIC_SHARPNESS_C) -> np.ndarray:
    """Correct slow brightness gradients and boost detail, working in the frequency domain.

    Model: image = illumination x reflectance. Illumination changes slowly (low
    frequencies), reflectance/detail changes quickly (high frequencies). Taking the
    log turns the product into a sum, so one filter can treat them separately:

        log -> FFT -> multiply by H(u, v) -> inverse FFT -> exp

    gamma_low < 1 flattens slow gradients across the abdomen; gamma_high > 1
    sharpens detail.

    Inputs:
        image: 2D uint8 slice.
        gamma_low, gamma_high, cutoff_d0, sharpness_c: see config (HOMOMORPHIC_*).
    Outputs:
        2D uint8 filtered image (rescaled to 0-255 with rescale_to_uint8).
    Reference:
        Oppenheim, A. V., Schafer, R. W., & Stockham, T. G. (1968). Nonlinear
        filtering of multiplied and convolved signals. Proceedings of the IEEE,
        56(8), 1264-1291.
        Gonzalez, R. C., & Woods, R. E. (2018). Digital image processing (4th ed.).
        Pearson. (Section on homomorphic filtering.)
    """
    # log(1 + x) instead of log(x): black air pixels are 0 and log(0) is undefined.
    log_image = np.log1p(image.astype(np.float64))

    # FFT, shifted so zero frequency is in the centre (matches how H is built).
    spectrum = np.fft.fftshift(np.fft.fft2(log_image))

    high_emphasis = gaussian_high_emphasis_filter(image.shape, gamma_low, gamma_high, cutoff_d0, sharpness_c)
    filtered_spectrum = spectrum * high_emphasis

    # Back to the image domain. The imaginary part is only rounding error (H is
    # symmetric), so we keep the real part.
    filtered_log_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)))

    # exp(x) - 1 undoes log(1 + x).
    filtered_image = np.expm1(filtered_log_image)
    return rescale_to_uint8(filtered_image)


# -----------------------------------------------------------------------------
# Final chain used by the pipeline
# -----------------------------------------------------------------------------
def enhance_before_contrast(hu_slice: np.ndarray) -> np.ndarray:
    """First three stages of the chain: window -> anisotropic diffusion -> homomorphic filter.

    Split out so step 3 can compute it once per slice and then try each contrast step.
    """
    windowed = apply_hu_window(hu_slice)
    denoised = anisotropic_diffusion_perona_malik(windowed)
    return homomorphic_filter(denoised)


def apply_contrast_step(image: np.ndarray, method: str = config.SEGMENTATION_CONTRAST) -> np.ndarray:
    """Last stage of the chain: "clahe", "histeq" (global histogram equalisation) or "none".

    Which one is used was decided by segmentation Dice on the training patients (step 3).
    """
    if method == "clahe":
        return clahe_enhancement(image)
    if method == "histeq":
        return histogram_equalisation(image)
    if method == "none":
        return image
    raise ValueError('contrast method must be "clahe", "histeq" or "none"')


def enhance_slice(hu_slice: np.ndarray, contrast_method: str = config.SEGMENTATION_CONTRAST) -> np.ndarray:
    """The enhancement chain the segmentation uses: window -> diffusion -> homomorphic -> contrast step.

    Why this order:
      1. Window first: every later step works on 8-bit grey levels, and the window
         decides which HU range those 256 levels describe.
      2. Anisotropic diffusion BEFORE the homomorphic filter: the homomorphic filter
         boosts high frequencies (gamma_high > 1), and CT noise is high-frequency.
         Denoising first means it sharpens organ edges, not noise.
      3. Homomorphic before the contrast step: it evens out slow brightness changes,
         so CLAHE's per-tile equalisation is not driven by them.
      4. Contrast step last (CLAHE / HE / none, chosen by segmentation Dice in step 3):
         it sets the final contrast the segmentation sees. Placed earlier, its noise
         amplification would be fed into the other filters.

    Inputs:
        hu_slice: 2D raw HU slice.
        contrast_method: "clahe", "histeq" or "none" (default from config).
    Outputs:
        2D uint8 enhanced image fed into Phase 2.
    """
    return apply_contrast_step(enhance_before_contrast(hu_slice), contrast_method)
