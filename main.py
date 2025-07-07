from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phishing_detector.pipeline import PhishingDetector

app = FastAPI()

# Allow the extension to connect to this API (CORS configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Chrome Extensions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the models
lr_detector = PhishingDetector(model_type="lr")
rf_detector = PhishingDetector(model_type="rf")

# Request schema
class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

@app.post("/predict")
async def predict(email: EmailRequest):
    # Process with both models
    lr_result = lr_detector.process_email(email.sender, email.subject, email.body)
    rf_result = rf_detector.process_email(email.sender, email.subject, email.body)

    # Ensemble prediction (same weighting you used in your project)
    LR_WEIGHT = 0.65
    RF_WEIGHT = 0.35
    combined_prob = (LR_WEIGHT * lr_result['phishing_probability'] + RF_WEIGHT * rf_result['phishing_probability']) / (LR_WEIGHT + RF_WEIGHT)

    is_phishing = combined_prob > 0.6
    confidence = 'High' if abs(combined_prob - 0.6) > 0.4 else 'Medium' if abs(combined_prob - 0.6) > 0.2 else 'Low'

    return {
        'is_phishing': is_phishing,
        'phishing_probability': combined_prob,
        'confidence': confidence
    }

@app.get("/")
def read_root():
    return {"status": "API is running"}
