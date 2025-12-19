import capture

if __name__ == "__main__":
    rawData = capture.captureRaw(
        exposureTime=20000,
        analogueGain=2.0,
        digitalGain=1.0
    )
    np.save("testImage.npy", rawData)
    print("Captured")
