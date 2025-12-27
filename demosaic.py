import numpy as np

def demosaic(raw, blackLevel=0, whiteBalance=None, colorMatrix=None, gamma=1.0):
    if raw.ndim != 2:
        raise ValueError("Raw input must be 2D array")
    if raw.dtype != np.uint16:
        raise ValueError("Raw input must be uint16")

    height, width = raw.shape

    data = raw.astype(np.float32)
    data = np.clip(data - blackLevel, 0, None)

    r = data[0::2, 0::2]
    g1 = data[0::2, 1::2]
    g2 = data[1::2, 0::2]
    b = data[1::2, 1::2]

    rFull = bilinearUpsample(r, height, width)
    g1Full = bilinearUpsample(g1, height, width)
    g2Full = bilinearUpsample(g2, height, width)
    g = (g1Full + g2Full) / 2
    bFull = bilinearUpsample(b, height, width)

    if whiteBalance is None:
        whiteBalance = [1.0, 1.0, 1.0]
    rFull *= whiteBalance[0]
    g *= whiteBalance[1]
    bFull *= whiteBalance[2]

    rgb = np.stack([rFull, g, bFull], axis=-1)

    if colorMatrix is not None:
        rgbFlat = rgb.reshape(-1, 3)
        rgb = (rgbFlat @ colorMatrix.T).reshape(height, width, 3)

    if gamma != 1.0:
        maxVal = 4095.0
        rgb = np.power(np.clip(rgb / maxVal, 0, 1), 1.0 / gamma) * maxVal

    return np.clip(rgb, 0, 4095).astype(np.uint16)

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

    rgb16 = (rgb.astype(np.float32) * (65535.0 / 4095.0)).astype(np.uint16)
    img = Image.fromarray(rgb16, mode="RGB")
    img.save(outputPath)
    print("Saved")
