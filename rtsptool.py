import os
import threading
import time
import datetime
import cv2
from urllib.parse import urlparse
import argparse
from astral.sun import sun
from astral import LocationInfo
from dateutil import tz
import logging
import tempfile
import shutil
import signal
import sys

capture_running = True  # Global flag to control the capturing loop

def signal_handler(sig, frame):
    global capture_running
    logging.info("Ctrl-C caught, stopping capture...")
    capture_running = False

def create_video_from_images(image_folder, video_path, four_cc, fps=25.0):
    logging.info(f"Creating video file")

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # Ensure images are in order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    extension = ".m4v"
    if four_cc.lower() == "mjpg":
        extension = ".avi"

    if not video_path.endswith(extension):
        video_path += extension

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*four_cc), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    logging.info(f"Video saved to {video_path}")

    # Remove images after creating the video
    for image in images:
        os.remove(os.path.join(image_folder, image))
    logging.info("Captured images have been removed.")

def compute_duration_for_nighttime(coordinates):
    """Compute capture duration from sunset to sunrise if it's nighttime.

    Args:
        coordinates (tuple, list, or str): A tuple, list, or string containing latitude and longitude.
                                           If a string, it should be in the format "latitude,longitude".

    Returns:
        float or None: The number of seconds from now until sunrise if it's nighttime, otherwise None.
    """
    # Check and parse coordinates based on input type
    if isinstance(coordinates, str):
        coordinates = tuple(map(float, coordinates.split(',')))
    elif isinstance(coordinates, (list, tuple)):
        if len(coordinates) != 2:
            raise ValueError("Coordinates must have exactly two elements (latitude and longitude).")
        coordinates = tuple(coordinates)  # Ensure coordinates are a tuple regardless of list or tuple input
    else:
        raise ValueError("Coordinates must be a tuple, a list, or a comma-separated string.")
    
    # Setting up the location with extracted latitude and longitude
    latitude, longitude = coordinates
    city = LocationInfo(name="", region="", latitude=str(latitude), longitude=str(longitude), timezone='UTC')
    
     # Get today's date and add one day to get tomorrow's date
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    # Calculating sunset today and sunrise tomorrow
    s_today = sun(city.observer, date=datetime.date.today())
    s_tomorrow = sun(city.observer, date=tomorrow)
    
    sunset_today = s_today['sunset'].astimezone(tz.tzlocal())
    sunrise_today = s_today['sunrise'].astimezone(tz.tzlocal())
    sunrise_tomorrow = s_tomorrow['sunrise'].astimezone(tz.tzlocal())

    # Determine if it's currently nighttime and compute duration accordingly
    now = datetime.datetime.now(tz.tzlocal())
    if now < sunset_today and now > sunrise_today:
        logging.info(f"Waiting for sunset at {sunset_today} to start capturing.")
        while now < sunset_today:
            time.sleep(60)
            now = datetime.datetime.now(tz.tzlocal())
    else:
        logging.info(f"It's night time, immediately starting to capture {now} {sunset_today} {sunrise_today}")
    
    return round((sunrise_tomorrow - now).total_seconds())

def capture_images(source, interval, quality, duration, output_directory, nighttime, coordinates, video, four_cc, width, height):
    """
    Capture images from a video source at specified intervals and optionally create a video from the captured images.

    Parameters:
    - source (int or str): The video source, typically a camera index or a video file path.
    - interval (float): Time interval (in seconds) between each image capture.
    - quality (int): JPEG quality of the saved images (0 to 100).
    - duration (float): Total duration (in seconds) for capturing images.
    - output_directory (str): Directory to save the captured images and video.
    - nighttime (bool): Flag indicating whether nighttime adjustments are needed.
    - coordinates (tuple): Geographic coordinates (latitude, longitude) for nighttime calculation.
    - video (bool): Flag indicating whether to create a video from the captured images.
    - four_cc (str): Four-character code for the video codec.
    - width (int): Width to resize the captured images and video frames. Use 0 to keep original width.
    - height (int): Height to resize the captured images and video frames. Use 0 to keep original height.
    
    Returns:
    - None
    """
    global capture_running
    capture_running = True

    temp_dir = tempfile.mkdtemp() if video else None
    start_datetime = datetime.datetime.now()
    capture_folder = temp_dir if video else os.path.join(output_directory, start_datetime.strftime('%Y%m%d%H%M%S'))

    if not os.path.exists(capture_folder):
        os.makedirs(capture_folder, exist_ok=True)

    if nighttime:
        duration = compute_duration_for_nighttime(coordinates)

    start_time = time.time()
    last_capture_time = start_time
    video_src = None

    while time.time() - start_time < duration and capture_running:
        current_time = time.time()
        if current_time - last_capture_time < interval:
            time.sleep(0.1)
            continue

        video_src = cv2.VideoCapture(source)
        retrieve, frame = video_src.read()
        if not retrieve:
            logging.warning("Failed to retrieve frame.")
        else:
            if (width*height>0):
                frame = cv2.resize(frame, (width, height))
            file_name = f'{datetime.datetime.now().strftime("%d%m%y-%H%M%S-%f")}.jpg'
            path = os.path.join(capture_folder, file_name)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            logging.debug(f"Captured image saved to {path}, {int(duration - (current_time - start_time))} seconds to go")
            last_capture_time = current_time

        video_src.release()
        time.sleep(max(0, interval - (time.time() - current_time)))  # Adjust sleep time if processing takes time

    if video and os.listdir(capture_folder):
        video_path = os.path.join(output_directory, f"timelapse_{start_datetime.strftime('%Y%m%d%H%M%S')}_{int(duration)}s")
        create_video_from_images(capture_folder, video_path, four_cc)
        shutil.rmtree(capture_folder)
        logging.info("Temporary image files have been removed after creating the video.")


def log_parameters(args):
    logging.debug("Capturing parameters:")
    logging.debug(f"  Source URL: {args.source}")
    logging.debug(f"  Interval: {args.interval} seconds")
    logging.debug(f"  JPEG Quality: {args.quality}")
    if (args.width*args.height>0):
        logging.debug(f"  Resize: Enabled")
        logging.debug(f"  Width: {args.width}")
        logging.debug(f"  Height: {args.height}")
    else: 
        logging.debug(f"  Resize: Disabled")

    if args.nighttime:
        logging.debug(f"  Nighttime Capture: {args.nights} nights")
        logging.debug(f"  Duration: will be calculated")
        logging.debug(f"  Coordinates: {args.coordinates}")
    else:
        logging.debug(f"  Duration: {args.duration} seconds")
    
    logging.debug(f"  Output Directory: {args.output_directory}")
    
    if args.video:
        logging.debug(f"  Video Creation: Enabled")
        logging.debug(f"  Video Codec: {args.fourcc}")
    else:
        logging.debug(f"  Video Creation: Disabled")


def main():
    parser = argparse.ArgumentParser(description="Capture images from a video source.")
    parser.add_argument("source", help="Video source URL")
    parser.add_argument("--interval", type=int, default=15, help="Interval between captures in seconds")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality of the captured images")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run the capture in seconds")
    parser.add_argument("--output_directory", type=str, default=os.path.realpath(os.path.dirname(__file__)), help="Directory to save captured images or final video")
    parser.add_argument("--nighttime", action='store_true', help="Enable capturing only from sunset to sunrise")
    parser.add_argument("--nights", default=1, type=int, help="Capture a given number of nights")
    parser.add_argument("--coordinates", type=str, default="41.222,-6.988", help="GPS coordinates for calculating sunrise and sunset times in format 'latitude,longitude'")
    parser.add_argument("--video", action='store_true', help="Create a video from captured images and store only the video")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="Video codec, choose between mp4v and MJPG")
    parser.add_argument("--verbose", action='store_true', help="Verbose logging")
    parser.add_argument("--width", default=0, type=int, help="Image width (if not set, images won't be resized)")
    parser.add_argument("--height", default=0, type=int, help="Image height (if not set, images won't be resized)")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.nighttime:
        loops=args.nights
    else:
        loops=1

    log_parameters(args)

    coordinates = None
    if args.coordinates:
        coordinates = tuple(map(float, args.coordinates.split(',')))

    if args.nighttime is not None and coordinates is None:
        logging.error("Coordinates must be defined for nighttime mode")
        sys.exit(1)

    for i in range(0,loops):
        capture_images(
            source=args.source,
            interval=args.interval,
            quality=args.quality,
            duration=args.duration,
            output_directory=args.output_directory,
            nighttime=args.nighttime,
            coordinates=coordinates,
            video=args.video,
            four_cc=args.fourcc,
            width=args.width,
            height=args.height,
        )
        time.sleep(1)

if __name__ == "__main__":
    main()