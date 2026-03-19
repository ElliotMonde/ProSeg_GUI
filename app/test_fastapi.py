import requests

url = "http://127.0.0.1:8000/segment"
files = [
    ("files", ("test1.dcm", b"dummy1", "application/dicom")),
    ("files", ("test2.dcm", b"dummy2", "application/dicom")),
]
try:
    response = requests.post(url, files=files)
    print("Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Exception:", e)
