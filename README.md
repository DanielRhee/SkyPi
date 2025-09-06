# SkyPi
A Raspberry Pi HQ Camera controller designed for Astrophotography

The library creates near-raw data from the Raspberry PI HQ Camera by capturing the CFA and applying color correction to the sensor reading. It then exports the file in the FITS format for scientific processing.

## Usage
It is recommended to use some form of a small dim external display to display the IP address of the local network and then use VNC to connect to the camera remotely. This allows control to be done finely and to decrease light pollution around the sensor. Other strategies such as using buttons to capture the image but this may introduce unnecessary vibration and movement on the camera which can be damaging during long exposures.
