import os
import httpx
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
from io import BytesIO
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import logging
import pytz
from datetime import datetime
import uvicorn
# from dask_cuda import LocalCUDACluster
# from dask.distributed import Client
from teamData import member_details

logging.basicConfig(filename="faceMatching.log", filemode='w')
logger = logging.getLogger("Face")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("faceMatching.log")
logger.addHandler(file_handler)
total_done = 0
total_error = 0

FACE_RECOGNITION_TOLERANCE = 0.8


def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


async def fetch_image(url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx errors
            return response.content
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


async def get_face_info_from_image(image_content):
    image = Image.open(BytesIO(image_content))
    image = np.array(image)
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        face_locations = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]
        top, right, bottom, left = face_locations[0]
        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        img_with_rectangles = Image.fromarray(image)
        draw = ImageDraw.Draw(img_with_rectangles)
        draw.rectangle([left-20, top-50, right+20, bottom], outline="green", width=5)
        return face_encoding, img_with_rectangles
    return None, None


async def compare_faces_from_url_with_local(image_url, folder_path, tolerance=FACE_RECOGNITION_TOLERANCE):
    image_content = await fetch_image(image_url)
    face_encoding_1, img_with_rectangles_1 = await get_face_info_from_image(image_content)
    best_matched_image = None
    best_accuracy = 0

    if face_encoding_1 is not None:
        tasks = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                local_image_path = os.path.join(folder_path, filename)
                tasks.append(process_image(local_image_path, face_encoding_1, tolerance))
        results = await asyncio.gather(*tasks)
        # Find the best matched image
        for filename, accuracy in results:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_matched_image = filename

    return best_matched_image


async def process_image(image_path, face_encoding_1, tolerance):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        match = face_recognition.compare_faces([face_encoding_1], face_encodings[0], tolerance=tolerance)[0]
        if match:
            face_distance = face_recognition.face_distance([face_encoding_1], face_encodings[0])[0]
            accuracy = 1 - face_distance
            return image_path, accuracy
    return None, 0


app = FastAPI()


class Item(BaseModel):
    url: str


@app.post("/tech")
async def create_items(items: Item):
    try:
        folder_path = "hdml_faces"
        matched_image = await compare_faces_from_url_with_local(items.url, folder_path)
        print(matched_image)
        if matched_image:
            name = os.path.splitext(os.path.basename(matched_image))[0]
            final_result = {name: member_details.get(name)}
            print(final_result)
            return {"data":final_result}
        else:
            return {"message":"Wrong Person !!!"}
            # raise HTTPException(status_code=404, detail="No matching image found")
    except Exception as e:
        global total_error
        total_error += 1
        logger.info(f"Time:{get_bd_time()}, Execution Failed and Total Failed Execution : {total_error}, Payload:{items}")
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        global total_done
        total_done += 1
        logger.info(f"Time:{get_bd_time()}, Execution Done and Total Successful Execution : {total_done}, Payload:{items}")


if __name__ == "__main__":
    try:
        # cluster = LocalCUDACluster()
        # client = Client(cluster)
        uvicorn.run(app, host="127.0.0.1", port=8060)
    except Exception as e:
        print(f"Server error: {str(e)}")
