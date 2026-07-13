import requests

response = requests.post(
    "http://0.0.0.0:8000/sleep-stage/predict",
    json={
        "record_id": '2026-02-21-Vy-4mm-10mA',
    }
)

result = response.json()
print(result)