import asyncio
import math
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from realtime.connection import Socket
from supabase import create_client, Client

load_dotenv()

# Supabase configuration

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']
IMAGE_TABLE_NAME = 'image'
DETECTIONS_TABLE_NAME = 'detection'
STORAGE_BUCKET = 'images'


def detect_hotpoints(image: np.ndarray) -> list[tuple[int, int]]:
    """Detects hotpoints in an image.

    Args:
        image: The image to process.

    Returns:
        A list of hotpoint coordinates, where each coordinate is a tuple (x, y).
    """

    # Example: Color Thresholding (adapt this to your specific hotpoint criteria)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Adjust threshold value

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotpoints = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        hotpoints.append((x + w // 2, y + h // 2))  # Center of the bounding box

    return hotpoints


def calculate_bounding_box(hotpoint: tuple[int, int]) -> tuple[int, int, int, int]:
    """Calculates a bounding box around a hotpoint.

    Args:
        hotpoint: The (x, y) coordinates of the hotpoint.

    Returns:
        A tuple (x, y, width, height) representing the bounding box.
    """

    x, y = hotpoint
    MARGIN = 20  # Adjust margin as needed

    return x - MARGIN, y - MARGIN, 2 * MARGIN, 2 * MARGIN


# Supabase Interaction Functions
async def process_image(db_image):
    image_id = db_image["id"]

    print("Processing:", db_image["id"])

    thermal = supabase.storage.from_(STORAGE_BUCKET).download(db_image["thermal_path"])

    image_array = np.frombuffer(thermal, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    hotpoints = detect_hotpoints(image)

    detections = list(map(lambda point: calculate_bounding_box(point), hotpoints))
    print("Storing", len(detections), "detections")

    await store_detections(db_image, detections)
    await mark_entry_processed(image_id)


async def store_detections(image, detections):
    for (x, y, w, h) in detections:
        supabase.table(DETECTIONS_TABLE_NAME).insert({
            'image': image["id"],
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'flight_date': image["flight_date"],
            'location': calculate_new_coordinates(image["location"]["longitude"], image["location"]["latitude"], 10, 30,
                                                  (320, 240), (x, y))
        }).execute()


async def mark_entry_processed(entry_id):
    supabase.table(IMAGE_TABLE_NAME).update({'processed': True}).eq('id', entry_id).execute()


async def process_db_update(db_image):
    image = db_image["record"]
    await process_image(image)


def calculate_new_coordinates(lon, lat, height, fov, img_dim, spot_pos):
    """
    Calculate new longitude and latitude of the spot.

    Parameters:
    - lon, lat: Longitude and latitude of the original position.
    - height: Height above the ground in meters.
    - fov: Field of view of the camera in degrees.
    - img_dim: A tuple (width, height) of the image in pixels.
    - spot_pos: A tuple (x, y) of the spot position in the image, from top-left corner.

    Returns:
    - Tuple (new_lon, new_lat): New longitude and latitude of the spot.
    """
    img_width, img_height = img_dim
    x, y = spot_pos

    # Assuming the middle of the image is the original coordinate
    dx_pixels = x - img_width / 2
    dy_pixels = img_height / 2 - y  # y is inverted in image coordinates

    # Calculate the scale of the image: meters per pixel
    # Approximation for small FOV and height
    scale = 2 * height * math.tan(math.radians(fov / 2)) / img_height

    # Calculate displacement in meters
    dx_meters = dx_pixels * scale
    dy_meters = dy_pixels * scale

    # Convert displacement to geographical coordinates
    # Approximation: 1 degree latitude = 111km, 1 degree longitude varies with latitude
    delta_lat = dy_meters / 111000  # Degrees
    delta_lon = dx_meters / (111000 * math.cos(math.radians(lat)))  # Degrees

    new_lat = lat + delta_lat
    new_lon = lon + delta_lon

    return {'latitude': new_lat, 'longitude': new_lon}


supabase: Client | None = None

if __name__ == "__main__":
    SUPABASE_ID = os.getenv("SUPABASE_URL").split("//")[1].split(".")[0]
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    supabase = create_client(os.getenv("SUPABASE_URL"), SUPABASE_KEY)

    URL = f"wss://{SUPABASE_ID}.supabase.co/realtime/v1/websocket?apikey={SUPABASE_KEY}&vsn=1.0.0"
    s = Socket(URL)
    s.connect()

    channel_1 = s.set_channel("realtime:public:image")
    channel_1.join().on("INSERT", lambda msg: asyncio.create_task(process_db_update(msg)))

    s.listen()

    # Also start listener that checks for unprocessed images periodically, if there was an error with the processing
    # for example
