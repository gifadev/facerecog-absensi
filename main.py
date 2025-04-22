import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List, Dict
import uvicorn
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from elasticsearch_utils import create_index
from face_recognition_utils import face_model, search_identity, register_identities, gen_frames
import json
import shutil
import os
from pydantic import BaseModel
from datetime import date
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional 

# Pydantic models untuk validasi data
class PersonalData(BaseModel):
    full_name: Optional[str] = None
    birth_place: Optional[str] = None
    birth_date: Optional[date] = None
    address: Optional[str] = None
    nationality: Optional[str] = None
    passport_number: Optional[str] = None
    gender: Optional[str] = None
    national_id_number: Optional[str] = None
    marital_status: Optional[str] = None
    phone_number: Optional[str] = None

class RegistrationData(BaseModel):
    image_paths: List[str]
    data: PersonalData

class RegisterResponse(BaseModel):
    message: str
    data: List[Dict]



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

import threading
import time
from face_recognition_utils import gen_frames

@app.on_event("startup")
def start_face_recognition():
    def worker():
        while True:
            try:
                for _ in gen_frames():
                    pass
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Face‑recog error:", e)
                time.sleep(1)

    t = threading.Thread(target=worker, daemon=True)
    t.start()


    # def worker(rtsp_url):
    #     while True:
    #         try:
    #             for _ in gen_frames(rtsp_url):
    #                 pass
    #         except Exception as e:
    #             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Face‑recog error ({rtsp_url}):", e)
    #             time.sleep(1)

    # cam1 = "rtsp://administrator:administrator@172.15.2.212:554/stream1"
    # cam2 = "rtsp://admin:Intek46835@172.15.5.59:554/stream/ch1"

    # t1 = threading.Thread(target=worker, args=(cam1,), daemon=True)
    # t2 = threading.Thread(target=worker, args=(cam2,), daemon=True)

    # t1.start()
    # t2.start()


app.mount("/regist", StaticFiles(directory="regist"), name="regist")
@app.post("/search/")
async def search(image: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"images/{image.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await image.read())
        
        # Load and process input image
        image = Image.open(file_path).convert("RGB")
        faces = face_model.get(np.array(image))

        if len(faces) == 0:
            print("No faces detected in the image.")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No faces detected in the image"
                }
            )
        
        # Call the search_identity function
        result = search_identity(file_path)
        
        if result:
            response = {
                "status": "success",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                "message": "data image",
                "data": result
            }
            return response
        else:
            print("No matching identity found")
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No matching identity found"
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

async def save_image(file: UploadFile, filename: str):
    try:
        os.makedirs("images", exist_ok=True)
        file_path = f"images/{filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

def clear_registration_file():
    with open('data_regist_api.json', 'w') as file:
        json.dump([], file)

@app.post("/register", response_model=RegisterResponse)
async def register(
    registration_data: str = Form(...),  # JSON string containing array of registration data
    images: List[UploadFile] = File(...)
):
    try:
        # Parse registration data JSON string
        registrations = json.loads(registration_data)
        
        if not isinstance(registrations, list):
            registrations = [registrations]

        # Dictionary to track images per person
        image_map = {}
        current_image_index = 0

        # Process each registration
        processed_registrations = []
        for reg in registrations:
            # Validate personal data
            personal_data = PersonalData(**reg['data'])
            
            # Calculate number of images needed for this person
            num_images_needed = len(reg['image_paths'])
            
            if current_image_index + num_images_needed > len(images):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Not enough images provided for {personal_data.full_name}"
                )

            # Save images for this person
            image_paths = []
            for i in range(num_images_needed):
                image = images[current_image_index + i]
                filename = f"{personal_data.full_name}_{i+1}{os.path.splitext(image.filename)[1]}"
                file_path = await save_image(image, filename)
                image_paths.append(file_path)

            current_image_index += num_images_needed

            # Create registration data
            processed_reg = {
                "image_paths": image_paths,
                "data": personal_data.model_dump()
            }
            processed_registrations.append(processed_reg)

        # Save to JSON file
        try:
            with open('data_regist_api.json', 'r') as file:
                existing_data = json.load(file)
                processed_registrations = existing_data + processed_registrations
        except FileNotFoundError:
            pass

        with open('data_regist_api.json', 'w') as file:
            json.dump(processed_registrations, file, indent=4, default=str)

        # Call register_identities with processed data
        register_identities(processed_registrations)

        # Clear the JSON file after successful registration
        clear_registration_file()

        return {
            "message": "Bulk registration successful",
            "data": processed_registrations
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    # Ensure Elasticsearch index exists
    create_index()

    uvicorn.run(app, host="0.0.0.0", port=8000)