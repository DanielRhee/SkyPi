#!/usr/bin/env python3
from picamera2 import Picamera2
import numpy as np

def unpackCsi12(packed, width, height, stride):
    # CSI-2 packed 12-bit: 2 pixels per 3 bytes, accounting for row stride
    bytesPerRow = (width * 3) // 2

    # Reshape to extract rows with stride, then trim to actual data
    packed2d = packed.reshape(height, stride)[:, :bytesPerRow].astype(np.uint16)

    byte0 = packed2d[:, 0::3]
    byte1 = packed2d[:, 1::3]
    byte2 = packed2d[:, 2::3]

    pixel0 = byte0 | ((byte2 & 0x0F) << 8)
    pixel1 = byte1 | ((byte2 & 0xF0) << 4)

    unpacked = np.empty((height, width), dtype=np.uint16)
    unpacked[:, 0::2] = pixel0
    unpacked[:, 1::2] = pixel1

    return unpacked

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

    # Unpack if needed (12-bit CSI-2 packed format)
    if raw.dtype == np.uint8:
        rawConfig = picam2.camera_configuration()["raw"]
        width = rawConfig["size"][0]
        height = rawConfig["size"][1]
        stride = rawConfig["stride"]
        raw = unpackCsi12(raw.flatten(), width, height, stride)

    return raw
