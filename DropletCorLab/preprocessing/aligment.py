import numpy as np
import cv2

def estimate_translation(image_ref_rgb: np.ndarray,
                         image_rgb: np.ndarray) -> tuple[int, int]:
    """
    Estimate translational shift between a reference image and a target image
    using phase correlation.

    :param
    ----------
    image_ref_rgb : np.ndarray
        Reference image in RGB format.
    image_rgb : np.ndarray
        Target image in RGB format.

    :return
    -------
    dx : int
        Horizontal shift (columns).
    dy : int
        Vertical shift (rows).

    Notes
    -----
    Positive dx shifts the image to the right.
    Positive dy shifts the image downward.
    """
    img1 = cv2.cvtColor(image_ref_rgb, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2).conj()

    cross_power = f1 * f2
    cross_power /= np.abs(cross_power) + 1e-12

    corr = np.fft.ifft2(cross_power)
    shift = np.unravel_index(np.argmax(np.abs(corr)), img1.shape)

    shift = np.array(shift)
    half = np.array(img1.shape) // 2
    shift[shift > half] -= np.array(img1.shape)[shift > half]

    dy, dx = shift
    return int(dx), int(dy)


def apply_translation(image_rgb: np.ndarray,
                      dx: int,
                      dy: int,
                      border_value=(255, 255, 255)) -> np.ndarray:
    """
    Apply a rigid translation to an image.

    :param
    ----------
    image_rgb : np.ndarray
        RGB image.
    dx, dy : int
        Translation in pixels.
    border_value : tuple
        Fill value for exposed borders.

    :return
    -------
    shifted_image : np.ndarray
    """
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])

    return cv2.warpAffine(
        image_rgb,
        M,
        (image_rgb.shape[1], image_rgb.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )


def histogram_match_rgb(source_rgb: np.ndarray,
                        reference_rgb: np.ndarray) -> np.ndarray:
    """
    Match the luminance distribution of source image to reference image
    using Lab color space.

    :return
    -------
    matched_rgb : np.ndarray
    """
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2Lab)
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2Lab)

    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, _, _ = cv2.split(ref_lab)

    src_hist, _ = np.histogram(src_l.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_l.flatten(), 256, [0, 256])

    src_cdf = np.cumsum(src_hist) / src_hist.sum()
    ref_cdf = np.cumsum(ref_hist) / ref_hist.sum()

    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lut[i] = j

    matched_l = lut[src_l]
    matched_lab = cv2.merge((matched_l, src_a, src_b))

    return cv2.cvtColor(matched_lab, cv2.COLOR_Lab2RGB)

def image_preprocessing(source_image: np.ndarray,
                        image_pos_ref: np.ndarray,
                        image_brightness_ref: np.ndarray) -> np.ndarray:
    """
    Match the position and brightness of source image to reference images

    :param
    ----------
    source_image: np.ndarray
        Image to be processed.
    image_pos_ref : np.ndarray
        Reference image for positional alignment.
    image_brightness_ref : np.ndarray
        Reference image for brightness alignment.

    :param
    -------
    aligned_ : np.ndarray
    """

    dx, dy = estimate_translation(image_pos_ref, source_image)
    pos_aligned = apply_translation(source_image, dx, dy)
    aligned_ = histogram_match_rgb(pos_aligned, image_brightness_ref)

    return aligned_
