import numpy as np
from scipy.ndimage import convolve

def demosaic(raw, blackLevel=256, whiteBalance=None, gamma=2.2, bayerPattern="BGGR"):
    if raw.ndim != 2:
        raise ValueError("Raw input must be 2D array")

    data = raw.astype(np.float32)
    data = np.clip(data - blackLevel, 0, None)
    data = data / (4095 - blackLevel)

    height, width = data.shape

    # Bayer pattern offsets: (r_row, r_col, b_row, b_col)
    patternOffsets = {
        "BGGR": (1, 1, 0, 0),
        "RGGB": (0, 0, 1, 1),
        "GBRG": (1, 0, 0, 1),
        "GRBG": (0, 1, 1, 0),
    }
    rRow, rCol, bRow, bCol = patternOffsets.get(bayerPattern, (1, 1, 0, 0))

    # Create color channel masks
    rMask = np.zeros((height, width), dtype=np.float32)
    gMask = np.zeros((height, width), dtype=np.float32)
    bMask = np.zeros((height, width), dtype=np.float32)

    rMask[rRow::2, rCol::2] = 1
    bMask[bRow::2, bCol::2] = 1
    gMask[1 - rRow::2, rCol::2] = 1
    gMask[rRow::2, 1 - rCol::2] = 1

    # Bilinear interpolation kernels
    rbKernel = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=np.float32) / 4.0

    gKernel = np.array([[0, 1, 0],
                        [1, 4, 1],
                        [0, 1, 0]], dtype=np.float32) / 4.0

    # Interpolate each channel with proper normalization for sparse data
    rWeight = convolve(rMask, rbKernel, mode='mirror')
    gWeight = convolve(gMask, gKernel, mode='mirror')
    bWeight = convolve(bMask, rbKernel, mode='mirror')

    r = convolve(data * rMask, rbKernel, mode='mirror') / np.maximum(rWeight, 1e-10)
    g = convolve(data * gMask, gKernel, mode='mirror') / np.maximum(gWeight, 1e-10)
    b = convolve(data * bMask, rbKernel, mode='mirror') / np.maximum(bWeight, 1e-10)

    if whiteBalance is None:
        whiteBalance = [1.9, 1.0, 1.4]
    r *= whiteBalance[0]
    g *= whiteBalance[1]
    b *= whiteBalance[2]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / gamma)

    return (rgb * 255).astype(np.uint8)

if __name__ == "__main__":
    import sys
    from PIL import Image

    inputPath = sys.argv[1]
    outputPath = sys.argv[2] if len(sys.argv) > 2 else inputPath.replace(".npy", "_demosaiced.png")
    bayerPattern = sys.argv[3] if len(sys.argv) > 3 else "BGGR"

    rawData = np.load(inputPath)
    rgb = demosaic(rawData, bayerPattern=bayerPattern)

    img = Image.fromarray(rgb, mode="RGB")
    img.save(outputPath)
    print("Saved")
