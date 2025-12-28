import numpy as np

def demosaic(raw, blackLevel=256, whiteBalance=None, gamma=2.2, autoStretch=True):
    # IMX477 uses BGGR pattern: B at [0,0], Gb at [0,1], Gr at [1,0], R at [1,1]
    if raw.ndim != 2:
        raise ValueError("Raw input must be 2D array")

    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16)

    height, width = raw.shape
    data = raw.astype(np.float32)
    data = np.clip(data - blackLevel, 0, None)

    # BGGR pattern extraction
    b = data[0::2, 0::2]
    gb = data[0::2, 1::2]
    gr = data[1::2, 0::2]
    r = data[1::2, 1::2]

    rFull = bilinearUpsample(r, height, width)
    gFull = (bilinearUpsample(gb, height, width) + bilinearUpsample(gr, height, width)) / 2
    bFull = bilinearUpsample(b, height, width)

    # Default white balance for daylight (IMX477 typical values)
    if whiteBalance is None:
        whiteBalance = [1.9, 1.0, 1.4]
    rFull *= whiteBalance[0]
    gFull *= whiteBalance[1]
    bFull *= whiteBalance[2]

    rgb = np.stack([rFull, gFull, bFull], axis=-1)

    # Auto-stretch to use full dynamic range
    if autoStretch:
        minVal = np.percentile(rgb, 1)
        maxVal = np.percentile(rgb, 99)
        rgb = (rgb - minVal) / (maxVal - minVal + 1e-6)
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb = rgb / (4095.0 - blackLevel)
        rgb = np.clip(rgb, 0, 1)

    # Apply gamma for display
    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / gamma)

    return (rgb * 255).astype(np.uint8)

def bilinearUpsample(channel, outputHeight, outputWidth):
    inputHeight, inputWidth = channel.shape

    y = np.linspace(0, inputHeight - 1, outputHeight)
    x = np.linspace(0, inputWidth - 1, outputWidth)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, inputHeight - 1)
    x1 = np.minimum(x0 + 1, inputWidth - 1)

    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)

    q00 = channel[y0[:, np.newaxis], x0[np.newaxis, :]]
    q01 = channel[y0[:, np.newaxis], x1[np.newaxis, :]]
    q10 = channel[y1[:, np.newaxis], x0[np.newaxis, :]]
    q11 = channel[y1[:, np.newaxis], x1[np.newaxis, :]]

    result = (q00 * (1 - wx) * (1 - wy) + q01 * wx * (1 - wy) + q10 * (1 - wx) * wy + q11 * wx * wy)

    return result.astype(np.float32)

if __name__ == "__main__":
    import sys
    from PIL import Image

    inputPath = sys.argv[1]
    outputPath = sys.argv[2] if len(sys.argv) > 2 else inputPath.replace(".npy", "_demosaiced.png")

    rawData = np.load(inputPath)
    rgb = demosaic(rawData)

    img = Image.fromarray(rgb, mode="RGB")
    img.save(outputPath)
    print("Saved")
