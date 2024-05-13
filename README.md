# RSTPTool 

RTSPTool is a simple tool to record timelapses from an RTSP video source. It is not designed for continuous recording as the video source is being opened 
and closed for every picture. Depending on your camera this can take some seconds which will limit the maximum capture rate.

## Installation

1. Install Python for your platform
2. Checkout the project
3. Install the required libraries using pip:
   pip install -r requirements.txt
4. Run it, check the command line options

## Video source URLs

URL starts with "rtsp://". You usually use username, password and IP address of the camera. Check the documentation of the manufacturer if you specifically 
have to enable it, how to set username, password and what the URL is.
Example:
rtsp://mycamuser:mycampassword@192.168.1.99
