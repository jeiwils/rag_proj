import requests

payload = {
    "prompt": "Test prompt",
    "n_predict": 5,
    "temperature": 0.7,
    "stop": None
}

r = requests.post("http://localhost:8000/completion", json=payload)
print(r.status_code, r.json())
