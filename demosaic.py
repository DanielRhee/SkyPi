import cv2
import numpy as np

def demosaic(raw, blackLevel=256, whiteBalance=None, gamma=2.2, bayerPattern="BGGR"):
    if raw.ndim != 2:
        raise ValueError("Raw input must be 2D array")

    data = raw.astype(np.float32)
    data = np.clip(data - blackLevel, 0, None)

    maxVal = 4095 - blackLevel
    data = (data / maxVal * 65535).astype(np.uint16)

    patterns = {
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
    }
    pattern = patterns.get(bayerPattern, cv2.COLOR_BayerBG2RGB)

    rgb = cv2.cvtColor(data, pattern)

    rgb = rgb.astype(np.float32) / 65535.0

    if whiteBalance is None:
        whiteBalance = [1.9, 1.0, 1.4]
    rgb[:, :, 0] *= whiteBalance[0]
    rgb[:, :, 1] *= whiteBalance[1]
    rgb[:, :, 2] *= whiteBalance[2]
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
