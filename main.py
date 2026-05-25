from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Healthcare Symptom Checker API")
app.mount("/static", StaticFiles(directory="static"), name="static")

class SymptomRequest(BaseModel):
    symptoms: list[str]

@app.get("/")
def interface():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: SymptomRequest):
    # TODO: call your ML / logic here
    return {
        "input_symptoms": data.symptoms,
        "predicted_disease": "Consult a doctor"
    }

