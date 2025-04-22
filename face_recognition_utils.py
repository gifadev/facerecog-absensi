import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import uuid
import os
from config import es, INDEX_NAME, BASE_URL,API_ABSEN_URL,HEADERS_ABSEN,API_WA_URL,HEADERS_WA
import cv2
from datetime import date, datetime
import time
import requests
import mysql.connector
from datetime import timedelta
import threading


# Initialize InsightFace model
face_model = FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_model.prepare(ctx_id=0)

os.makedirs("regist", exist_ok=True)


def calculate_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "namaku",
    "database": "db_absensi",
}

def wa_sent_today(name: str) -> bool:
    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor()
    today = date.today().isoformat()
    cur.execute(
        "SELECT 1 FROM absensis "
        "WHERE nama=%s AND DATE(waktu_absen)=%s AND status_send_wa=1 LIMIT 1",
        (name, today)
    )
    sent = cur.fetchone() is not None
    cur.close()
    conn.close()
    return sent

def should_send_absen(name: str) -> bool:
    """
    Cek di DB: apakah tidak ada record absensi untuk `name`
    dalam 5 menit terakhir?
    Kembalikan True kalau boleh kirim (tidak ada record),
    False kalau harus skip.
    """
    now_dt      = datetime.now()
    cutoff_time = now_dt - timedelta(minutes=5)

    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute(
        "SELECT 1 FROM absensis WHERE nama=%s AND waktu_absen >= %s LIMIT 1",
        (name, cutoff_time)
    )
    found = cur.fetchone() is not None
    cur.close()
    conn.close()

    # kalau ditemukan record, berarti masih dalam 5 menit → jangan kirim
    return not found

def mark_wa_sent(record_id: int):
    """Tandai status_send_wa = true di record dengan id."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute(
        "UPDATE absensis SET status_send_wa=1 WHERE id=%s",
        (record_id,)
    )
    conn.commit()
    cur.close()
    conn.close()


# def save_face_image(image: np.array, face: dict, output_path: str):
#     x1, y1, x2, y2 = face['bbox'].astype(int)

#     face_image = image[y1:y2, x1:x2]

#     Image.fromarray(face_image).save(output_path)

def save_face_image(image: np.ndarray, face, output_path: str):
    # 1. Ambil bbox dengan properti face.bbox
    x1, y1, x2, y2 = face.bbox.astype(int)

    # 2. Clamp ke dalam ukuran image
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # 3. Skip jika crop menghasilkan area kosong
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid bbox {face.bbox}, skipping save")
        return

    # 4. Crop dan simpan
    face_img = image[y1:y2, x1:x2]
    Image.fromarray(face_img).save(output_path)


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

            #get max bbox (for get closest face)
            face = max(faces, key=lambda face: calculate_bbox_area(face.bbox))

            # Assuming single face per image
            # face = face[0]
            
            embedding = face.embedding.tolist()

            file_extension = os.path.splitext(image_path)[1]  
            random_name = f"{uuid.uuid4().hex}{file_extension}"  
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
                "marital_status": data.get("marital_status", ""),
                "phone_number": data.get("phone_number", "")
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
    
    #get max bbox (for get closest face)
    face = max(faces, key=lambda face: calculate_bbox_area(face.bbox))

    # Assuming single face per image
    # face = face[0]  

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

def search_identity_by_embedding(query_embedding, threshold=0.3):
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
        score = top_hit["_score"] - 1.0  # menyesuaikan perhitungan
        if score >= threshold:
            data = top_hit['_source']['data']
            return {
                "score": score,
                "data": data
            }
    return None


def gen_frames(sample_rate=5):
    rtsp_url = "rtsp://administrator:administrator@172.15.2.212:554/stream1"
    # rtsp_url = "rtsp://admin:Intek46835@172.15.5.59:554/stream/ch1"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat mengakses camera")

    frame_idx = 0
    last_result = None

    # Siapkan counter untuk FPS aktual (opsional)
    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Stream berakhir atau gagal baca frame")
                break

            frame_idx   += 1
            frame_count += 1

            # Hitung FPS aktual setiap detik
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                actual_fps = frame_count / elapsed
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Actual FPS: {actual_fps:.2f}")
                frame_count = 0
                fps_start   = time.time()

            # Deteksi wajah
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces     = face_model.get(rgb_frame)
            now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if faces and frame_idx % sample_rate == 0:
                for face in faces:
                    try:
                        result = search_identity_by_embedding(face.embedding)
                        if not result:
                            print(f"{now} – Detected: Unknown")
                            continue

                        name         = result["data"].get("full_name", "Unknown")
                        phone_number = result["data"].get("phone_number")
                        score        = result["score"] 
                        print(f"{now} – Detected: {name} with score : {score}")

                        # # 1) Crop & encode
                        # x1, y1, x2, y2 = map(int, face.bbox)
                        # h, w = frame.shape[:2]
                        # x1, y1 = max(0, x1), max(0, y1)
                        # x2, y2 = min(w, x2), min(h, y2)
                        # if x2 <= x1 or y2 <= y1:
                        #     print("Invalid bbox, skip")
                        #     continue

                        # face_crop = frame[y1:y2, x1:x2]
                        # success_enc, buf = cv2.imencode('.jpg', face_crop)
                        # if not success_enc:
                        #     print("Gagal encode foto wajah")
                        #     continue
                        # jpg_bytes = buf.tobytes()

                        
                        # setelah dapat face.bbox:
                        if not should_send_absen(name):
                            print(f"{now} – Skip absensi untuk {name} with score {score}, sudah dalam 5 menit terakhir")
                        else:
                            x1, y1, x2, y2 = map(int, face.bbox)
                            h, w = frame.shape[:2]

                            # 1) Hitung ukuran bbox
                            face_w = x2 - x1
                            face_h = y2 - y1

                            # 2) Tentukan padding
                            pad_w = int(face_w * 0.5)    # 50% lebar wajah
                            pad_h_top    = int(face_h * 0.5)  # 50% tinggi wajah ke atas
                            pad_h_bottom = int(face_h * 0.5)  # 50% tinggi wajah ke bawah untuk dada

                            # 3) Perluas bbox
                            x1_new = max(0, x1 - pad_w)
                            y1_new = max(0, y1 - pad_h_top)
                            x2_new = min(w, x2 + pad_w)
                            y2_new = min(h, y2 + pad_h_bottom)

                            # 4) Pastikan bbox valid
                            if x2_new <= x1_new or y2_new <= y1_new:
                                print("Invalid expanded bbox, skip")
                                continue

                            # 5) Crop dengan bbox baru
                            face_crop = frame[y1_new:y2_new, x1_new:x2_new]

                            # 6) Encode dan kirim
                            success_enc, buf = cv2.imencode('.jpg', face_crop)
                            if not success_enc:
                                print("Gagal encode foto wajah")
                                continue
                            jpg_bytes = buf.tobytes()
                            # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                            try:
                                resp = requests.post(
                                    API_ABSEN_URL,
                                    headers=HEADERS_ABSEN,
                                    data={
                                            "nama":  name,
                                            "score": score
                                        },
                                    files={"foto": ("face.jpg", jpg_bytes, "image/jpeg")}
                                )
                                print("→ Absensi:", resp.status_code, resp.ok)
                            except Exception as e:
                                print("Error kirim absensi:", e)
                                continue

                            # Ambil record ID dari response
                            rec_id = None
                            if resp.ok:
                                rec_json = resp.json()
                                rec_id   = rec_json.get("id") or rec_json.get("data",{}).get("id")

                            # 2) Cek dan kirim WA sekali per hari
                            if phone_number and phone_number != "-" and rec_id and not wa_sent_today(name):
                                wa_payload = {
                                    "to":      f"{phone_number}@s.whatsapp.net",
                                    "message": f"{now} – Wajah terdeteksi: {name}"
                                }
                                try:
                                    resp_wa = requests.post(API_WA_URL, headers=HEADERS_WA, json=wa_payload)
                                    print("→ WA:", resp_wa.status_code, resp_wa.ok)
                                    if resp_wa.ok:
                                        mark_wa_sent(rec_id)
                                except Exception as e:
                                    print("Error kirim WA:", e)

                    except Exception as ex:
                        print(f"Error proses face: {ex}")
                        continue

            # Jeda singkat agar CPU tidak full
            time.sleep(0.001)

    finally:
        cap.release()

