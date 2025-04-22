from elasticsearch import Elasticsearch

# Konfigurasi Elasticsearch
es = Elasticsearch(
    "http://157.245.200.237:9200",
    http_auth=("admin", "4dm1nus3r")
)
INDEX_NAME = "insightface"
BASE_URL = "http://157.245.200.237:8000"
API_ABSEN_URL = "http://172.15.0.122:80/api/absensi"
HEADERS_ABSEN = {"Accept": "application/json"}
API_WA_URL    = "http://172.15.3.173:8080/send-message"
HEADERS_WA    = {"Content-Type": "application/json"}
