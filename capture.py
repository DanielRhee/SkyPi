#!/usr/bin/env python3
from picamera2 import Picamera2
import numpy as np

def captureRaw(exposureTime=10000, analogueGain=1.0, focus=None, sharpness=None):
    picam2 = Picamera2()
    config = picam2.create_still_configuration(raw={})
    picam2.configure(config)

    controls = {
        "ExposureTime": exposureTime,
        "AnalogueGain": analogueGain,
    }

    if focus is not None:
        controls["AfMode"] = 0
        controls["LensPosition"] = focus
    if sharpness is not None:
        controls["Sharpness"] = sharpness

    picam2.set_controls(controls)
    picam2.start()
    raw = picam2.capture_array("raw")
    picam2.stop()

    return raw
