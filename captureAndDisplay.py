import argparse
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import capture
from demosaic import demosaic

def main():
    parser = argparse.ArgumentParser(description="Capture image, save as .npy, and display")
    parser.add_argument("-e", "--exposure", type=int, default=10000, help="Exposure time in microseconds")
    parser.add_argument("-a", "--analogue-gain", type=float, default=1.0, help="Analogue gain")
    parser.add_argument("-f", "--focus", type=float, default=None, help="Lens position / focus")
    parser.add_argument("-s", "--sharpness", type=float, default=None, help="Sharpness")
    parser.add_argument("-o", "--output-dir", type=str, default=".", help="Output directory for .npy file")
    parser.add_argument("-b", "--bayer", type=str, default="BGGR",
                        choices=["BGGR", "RGGB", "GBRG", "GRBG"],
                        help="Bayer pattern for demosaicing")
    args = parser.parse_args()

    rawData = capture.captureRaw(
        exposureTime=args.exposure,
        analogueGain=args.analogue_gain,
        focus=args.focus,
        sharpness=args.sharpness
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_dir, f"{timestamp}.npy")
    np.save(filename, rawData)
    print(f"Saved: {filename}")

    rgb = demosaic(rawData, bayerPattern=args.bayer)

    plt.imshow(rgb)
    plt.title(f"Capture: {timestamp}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
