import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from elasticsearch import Elasticsearch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
import os
from fastapi.responses import JSONResponse

# Initialize Elasticsearch client
es = Elasticsearch(
    "http://localhost:9200",
    http_auth=("admin", "4dm1nus3r")
)
INDEX_NAME = "insightface"
BASE_URL = "http://172.15.3.237:8000"
# Initialize InsightFace model
face_model = FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_model.prepare(ctx_id=0)  # Use GPU if available, otherwise use CPU

# Initialize FastAPI app
app = FastAPI()
os.makedirs("regist", exist_ok=True)

# Serve static files (arahkan ke folder images yang sudah ada)
app.mount("/regist", StaticFiles(directory="regist"), name="regist")

def create_index():
    if not es.indices.exists(index=INDEX_NAME):
        mapping = {
            "mappings": {
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "full_name": {"type": "keyword"},
                            "birth_place": {"type": "keyword"},
                            "birth_date": {"type": "date"},
                            "address": {"type": "text"},
                            "nationality": {"type": "keyword"},
                            "passport_number": {"type": "keyword"},
                            "gender": {"type": "keyword"},
                            "national_id_number": {"type": "keyword"},
                            "marital_status": {"type": "keyword"}
                        }
                    },
                    "photos": {
                        "type": "nested",
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": 512},
                            "image": {"type": "keyword"}
                        }
                    }
                }
            }
        }
        es.indices.create(index=INDEX_NAME, body=mapping)

def save_face_image(image: np.array, face: dict, output_path: str):
    # Ambil bounding box wajah
    x1, y1, x2, y2 = face['bbox'].astype(int)
    
    # Potong wajah dari gambar
    face_image = image[y1:y2, x1:x2]
    
    # Simpan wajah ke file
    Image.fromarray(face_image).save(output_path)

def register_identities(identities):
    for identity in identities:
        image_paths = identity["image_paths"]
        data = identity["data"]
        photos = []

        for image_path in image_paths:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            faces = face_model.get(image_np)

            if len(faces) == 0:
                print(f"No faces detected in the image: {image_path}")
                continue

            face = faces[0]  # Assuming single face per image
            embedding = face.embedding.tolist()

            # Generate nama acak dengan ekstensi file asli
            file_extension = os.path.splitext(image_path)[1]  # Ambil ekstensi file
            random_name = f"{uuid.uuid4().hex}{file_extension}"  # Nama acak + ekstensi
            new_file_path = f"regist/{random_name}"

            # Simpan wajah yang terdeteksi ke folder regist
            save_face_image(image_np, face, new_file_path)

            photos.append({
                "embedding": embedding,
                "image": new_file_path
            })

        if not photos:
            print(f"No valid photos to register for {data.get('full_name', 'Unknown')}.")
            continue

        # Save to Elasticsearch
        doc = {
            "data": {
                "full_name": data.get("full_name", ""),
                "birth_place": data.get("birth_place", ""),
                "birth_date": data.get("birth_date", ""),
                "address": data.get("address", ""),
                "nationality": data.get("nationality", ""),
                "passport_number": data.get("passport_number", ""),
                "gender": data.get("gender", ""),
                "national_id_number": data.get("national_id_number", ""),
                "marital_status": data.get("marital_status", "")
            },
            "photos": photos,
        }
        es.index(index=INDEX_NAME, document=doc)
        print(f"Registered {data.get('full_name', 'Unknown')} successfully")


def search_identity(image_path, threshold=0.3):
    # Load and process input image
    image = Image.open(image_path).convert("RGB")
    faces = face_model.get(np.array(image))

    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    face = faces[0]  # Assuming single face per image
    query_embedding = np.array(face.embedding)

    # Query Elasticsearch
    script_query = {
        "nested": {
            "path": "photos",
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'photos.embedding') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            }
        }
    }

    response = es.search(index=INDEX_NAME, query=script_query, size=1)
    hits = response["hits"]["hits"]

    if hits:
        top_hit = hits[0]
        score = top_hit["_score"] - 1.0  # Adjust for cosine similarity
        if score >= threshold:
            data = top_hit['_source']['data']
            matching_photo = top_hit['_source']['photos'][0]['image']
            # for photo in top_hit['_source']['photos']:
            #     if np.allclose(query_embedding, np.array(photo['embedding']), atol=1e-2):
            #         matching_photo = photo['image']
            #         break

            print(f"Match found: {data['full_name']} with image {matching_photo} and similarity {score}")
            print("Details:")
            print(f"Full Name: {data['full_name']}")
            print(f"Birth Place: {data['birth_place']}")
            print(f"Birth Date: {data['birth_date']}")
            print(f"Address: {data['address']}")
            print(f"Nationality: {data['nationality']}")
            print(f"Passport Number: {data['passport_number']}")
            print(f"Gender: {data['gender']}")
            print(f"National ID Number: {data['national_id_number']}")
            print(f"Marital Status: {data['marital_status']}")
            print(matching_photo)
            return {
                "image_url": f"{BASE_URL}/regist/{matching_photo.split('/')[-1]}",
                "full_name": data['full_name'],
                "birth_place": data['birth_place'],
                "birth_date": data['birth_date'],
                "address": data['address'],
                "nationality": data['nationality'],
                "passport_number": data['passport_number'],
                "gender": data['gender'],
                "national_id_number": data['national_id_number'],
                "marital_status": data['marital_status'],
                "score": score
            }

    print("No matching identity found.")
    return None

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
    
if __name__ == "__main__":
    # Ensure Elasticsearch index exists
    create_index()

    identities = [
        {
            "image_paths": [
                "images/Arif.jpg",
                "images/Arif 2.jpg",
                "images/Arif 3.jpg",
                "images/Arif 4.jpg",
                "images/Arif 5.jpg",
            ],
            "data": {
                "full_name": "Arif",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Arrizque.jpg",
                "images/Arrizque 2.jpg",
                "images/Arrizque 3.jpg",
            ],
            "data": {
                "full_name": "Arrizque",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Azhar.jpg",
                "images/Azhar 2.jpg",
                "images/Azhar 3.jpg",
                "images/Azhar 4.jpg",
            ],
            "data": {
                "full_name": "Azhar",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Chairul.jpg",
                "images/Chairul 2.jpg",
                "images/Chairul 3.jpg",
                "images/Chairul 4.jpg",
                "images/Chairul 5.jpg",
            ],
            "data": {
                "full_name": "Chairul",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Chandra.jpg",
                "images/Chandra 2.jpg",
                "images/Chandra 3.jpg",
                "images/Chandra 4.jpg",
                "images/Chandra 5.jpg",
            ],
            "data": {
                "full_name": "Chandra",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Christoper.jpg",
                "images/Christoper 2.jpg",
                "images/Christoper 3.jpg",
            ],
            "data": {
                "full_name": "Christoper",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Eko.jpg",
                "images/Eko 2.jpg",
                "images/Eko 3.jpg",
            ],
            "data": {
                "full_name": "Eko",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Fajri.jpg",
                "images/Fajri 2.jpg",
                "images/Fajri 3.jpg",
                "images/Fajri 4.jpg",
                "images/Fajri 5.jpg",
            ],
            "data": {
                "full_name": "Fajri",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Fandi.jpg",
                "images/Fandi 2.jpg",
                "images/Fandi 3.jpg",
                "images/Fandi 4.jpg",
            ],
            "data": {
                "full_name": "Fandi",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Ferdi.jpg",
                "images/Ferdi 2.jpg",
                "images/Ferdi 3.jpg",
                "images/Ferdi 4.jpg",
                "images/Ferdi 5.jpg",
            ],
            "data": {
                "full_name": "Ferdi",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Gisna.jpg",
                "images/Gisna 2.jpg",
                "images/Gisna 3.jpg",
                "images/Gisna 4.jpg",
                "images/Gisna 5.jpg",
            ],
            "data": {
                "full_name": "Gisna",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Hanif.jpg",
                "images/Hanif 2.jpg",
                "images/Hanif 3.jpg",
            ],
            "data": {
                "full_name": "Hanif",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Jaki.jpg",
                "images/Jaki 2.jpg",
                "images/Jaki 3.jpg",
                "images/Jaki 4.jpg",
                "images/Jaki 5.jpg",
            ],
            "data": {
                "full_name": "Jaki",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Kharisma.jpg",
                "images/Kharisma 2.jpg",
                "images/Kharisma 3.jpg",
            ],
            "data": {
                "full_name": "Kharisma",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Kunto.jpg",
                "images/Kunto 2.jpg",
                "images/Kunto 3.jpg",
            ],
            "data": {
                "full_name": "Kunto",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Lutfi.jpg",
                "images/Lutfi 2.jpg",
                "images/Lutfi 3.jpg",
                "images/Lutfi 4.jpg",
                "images/Lutfi 5.jpg",
            ],
            "data": {
                "full_name": "Lutfi",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Parlin.jpg",
                "images/Parlin 2.jpg",
                "images/Parlin 3.jpg",
                "images/Parlin 4.jpg",
                "images/Parlin 5.jpg",
            ],
            "data": {
                "full_name": "Parlin",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Radit.jpg",
                "images/Radit 2.jpg",
                "images/Radit 3.jpg",
            ],
            "data": {
                "full_name": "Radit",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Rajadi.jpg",
                "images/Rajadi 2.jpg",
                "images/Rajadi 3.jpg",
                "images/Rajadi 4.jpg",
            ],
            "data": {
                "full_name": "Rajadi",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Reyhan.jpg",
                "images/Reyhan 2.jpg",
                "images/Reyhan 3.jpg",
            ],
            "data": {
                "full_name": "Reyhan",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Ridzki.jpg",
                "images/Ridzki 2.jpg",
                "images/Ridzki 3.jpg",
            ],
            "data": {
                "full_name": "Ridzki",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Rizki.jpg",
                "images/Rizki 2.jpg",
                "images/Rizki 3.jpg",
                "images/Rizki 4.jpg",
                "images/Rizki 5.jpg",
                "images/Rizki 6.jpg",
            ],
            "data": {
                "full_name": "Rizki",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Rizky.jpg",
                "images/Rizky 2.jpg",
                "images/Rizky 3.jpg",
                "images/Rizky 4.jpg",
                "images/Rizky 5.jpg",
            ],
            "data": {
                "full_name": "Rizky",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Somad.jpg",
                "images/Somad 2.jpg",
                "images/Somad 3.jpg",
                "images/Somad 4.jpg",
                "images/Somad 5.jpg",
            ],
            "data": {
                "full_name": "Somad",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Tora.jpg",
                "images/Tora 2.jpg",
                "images/Tora 3.jpg",
            ],
            "data": {
                "full_name": "Tora",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Wahyu.jpg",
                "images/Wahyu 2.jpg",
                "images/Wahyu 3.jpg",
                "images/Wahyu 4.jpg",
                "images/Wahyu 5.jpg",
            ],
            "data": {
                "full_name": "Wahyu",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Wildan.jpg",
                "images/Wildan 2.jpg",
                "images/Wildan 3.jpg",
                "images/Wildan 4.jpg",
                "images/Wildan 5.jpg",
            ],
            "data": {
                "full_name": "Wildan",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Yasser.jpg",
                "images/Yasser 2.jpg",
            ],
            "data": {
                "full_name": "Yasser",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Yusuf.jpg",
                "images/Yusuf 2.jpg",
                "images/Yusuf 3.jpg",
                "images/Yusuf 4.jpg",
                "images/Yusuf 5.jpg",
            ],
            "data": {
                "full_name": "Yusuf",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        {
            "image_paths": [
                "images/Zagoto.jpg",
                "images/Zagoto 2.jpg",
                "images/Zagoto 3.jpg",
                "images/Zagoto 4.jpg",
                "images/Zagoto 5.jpg",
            ],
            "data": {
                "full_name": "Zagoto",
                "birth_place": "Jakarta",
                "birth_date": "1990-01-01",
                "address": "Jl. Merdeka No. 123, Jakarta",
                "nationality": "Indonesian",
                "passport_number": "A12345678",
                "gender": "Male",
                "national_id_number": "1234567890123456",
                "marital_status": "Single"
            }
        },
        
    ]

    # Register all identities
    # register_identities(identities)

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)