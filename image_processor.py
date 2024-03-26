import asyncio
import os
from typing import List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from realtime.connection import Socket
from supabase import create_client, Client
from abc import ABC, abstractmethod

load_dotenv()

# Supabase configuration

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']
IMAGE_TABLE_NAME = 'image'
DETECTIONS_TABLE_NAME = 'detection'
STORAGE_BUCKET = 'images'


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        pass


class HotPointDetector(Detector):
    def __init__(self):
        pass

    def _detect_hotpoints(self, image: np.ndarray) -> list[tuple[int, int]]:
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

    def _calculate_bounding_box(self, hotpoint: tuple[int, int]) -> tuple[int, int, int, int]:
        """Calculates a bounding box around a hotpoint.

        Args:
            hotpoint: The (x, y) coordinates of the hotpoint.

        Returns:
            A tuple (x, y, width, height) representing the bounding box.
        """

        x, y = hotpoint
        MARGIN = 20  # Adjust margin as needed

        return x - MARGIN, y - MARGIN, 2 * MARGIN, 2 * MARGIN

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        hotpoints = self._detect_hotpoints(image)

        detections = list(map(lambda point: self._calculate_bounding_box(point), hotpoints))
        return detections


async def process_image(db_image):
    image_id = db_image["id"]

    print("Processing:", db_image["id"])

    thermal = supabase.storage.from_(STORAGE_BUCKET).download(db_image["thermal_path"])

    image_array = np.frombuffer(thermal, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    detector = HotPointDetector()

    detections = detector.detect(image)

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
            'flight_date': image["flight_date"]
        }).execute()


async def mark_entry_processed(entry_id):
    supabase.table(IMAGE_TABLE_NAME).update({'processed': True}).eq('id', entry_id).execute()


async def process_db_update(db_image):
    image = db_image["record"]
    await process_image(image)


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
