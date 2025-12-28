import capture
import numpy

if __name__ == "__main__":
    rawData = capture.captureRaw(
        exposureTime=20000,
        analogueGain=2.0
    )
    np.save("testImage.npy", rawData)
    print("Captured")
