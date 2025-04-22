from config import es, INDEX_NAME

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
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
        