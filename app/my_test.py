from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()

@app.post("/test")
async def test(files: List[UploadFile]):
    return {"files": [f.filename for f in files]}
