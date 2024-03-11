import asyncio
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
async def process_image(image):
    image_id = image["id"]

    rgb = supabase.storage.from_(STORAGE_BUCKET).download(image["rgb_path"])
    thermal = supabase.storage.from_(STORAGE_BUCKET).download(image["thermal_path"])

    image_array = np.frombuffer(rgb, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    hotpoints = detect_hotpoints(image)

    for hotpoint in hotpoints:
        x, y, w, h = calculate_bounding_box(hotpoint)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    await store_detections(image_id, hotpoints)
    await mark_entry_processed(image_id)


async def store_detections(image_id, hotpoints):
    detections = []
    for (x, y, w, h) in hotpoints:
        detections.append({
            'image_id': image_id,
            'x': x,
            'y': y,
            'width': w,
            'height': h
        })
    await supabase.table(DETECTIONS_TABLE_NAME).insert(detections).execute()


async def mark_entry_processed(entry_id):
    await supabase.table(IMAGE_TABLE_NAME).update({'processed': True}).eq('id', entry_id).execute()


def process_db_update(db_image):
    image = db_image["record"]
    asyncio.run(process_image(image))


supabase: Client | None = None

if __name__ == "__main__":
    SUPABASE_ID = os.getenv("SUPABASE_URL").split("//")[1].split(".")[0]
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    supabase = create_client(os.getenv("SUPABASE_URL"), SUPABASE_KEY)

    URL = f"wss://{SUPABASE_ID}.supabase.co/realtime/v1/websocket?apikey={SUPABASE_KEY}&vsn=1.0.0"
    s = Socket(URL)
    s.connect()

    channel_1 = s.set_channel("realtime:public:image")
    channel_1.join().on("INSERT", process_db_update)

    s.listen()