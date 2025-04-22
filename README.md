# Face Recognition and Identity Search System

Proyek ini adalah sistem pengenalan wajah dan pencarian identitas yang menggunakan **InsightFace** untuk ekstraksi fitur wajah dan **Elasticsearch** untuk penyimpanan dan pencarian data. Sistem ini dibangun dengan **FastAPI** sebagai framework backend.

## Fitur
1. **Registrasi Identitas**: Mendaftarkan identitas beserta foto wajah ke dalam Elasticsearch.
2. **Pencarian Identitas**: Mencari identitas berdasarkan foto wajah yang diunggah.

## Teknologi yang Digunakan
- **InsightFace**: Untuk deteksi dan ekstraksi fitur wajah.
- **Elasticsearch**: Untuk penyimpanan dan pencarian data identitas.
- **FastAPI**: Framework untuk membangun API.
- **Uvicorn**: Server untuk menjalankan aplikasi FastAPI.

## Instalasi

### Prasyarat
1. **Python 3.8 atau lebih baru**.
2. **Elasticsearch** terinstal dan berjalan di `http://localhost:9200`.

### Langkah-langkah
1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/gifadev/face-recognation-insightface.git
   cd face-recognation-insightface
2. **Buat dan aktifkan virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/MacOS
    venv\Scripts\activate     # Untuk Windows
3. **Instal dependensi**:
    ```bash
    pip install -r requirements.txt
4. **Jalkan Elasticsearch**:

5. **Jalankan aplikasi:**:
    ```bash
    python main.py