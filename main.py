from fastapi import FastAPI, Form, HTTPException
import requests
import json
import logging
from dotenv import load_dotenv
import os
import pytesseract
import cv2
import numpy as np
from google import genai
import uvicorn
import re

# =====================================================
# CONFIG
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="OCR + Gemini API")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY manquante")

# ✅ BON CLIENT
client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# OCR
# =====================================================

def preprocess_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # amélioration contraste
    clahe = cv2.createCLAHE(clipLimit=3.0)
    gray = clahe.apply(gray)

    # débruitage
    gray = cv2.fastNlMeansDenoising(gray)

    # threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


def extract_text(image_bytes: bytes):
    try:
        img = preprocess_image(image_bytes)

        text = pytesseract.image_to_string(
            img,
            lang="fra+eng",
            config="--oem 3 --psm 6"
        )

        text = re.sub(r'\s+', ' ', text).strip()

        logger.info(f"OCR OK ({len(text)} chars)")

        return text

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


# =====================================================
# GEMINI
# =====================================================

def clean_json(text: str):
    return text.replace("```json", "").replace("```", "").strip()


def parse_amount(value):
    if not value:
        return None

    value = re.sub(r'[^\d,.\-]', '', str(value))
    value = value.replace(',', '.')

    try:
        return float(value)
    except:
        return None


def analyze_with_gemini(text: str):
    prompt = f"""
Analyse ce texte OCR de facture.

Retourne STRICTEMENT JSON :

{{
  "type_document": "invoice|receipt|other",
  "category": "telecom|restaurant|utilities|other",
  "supplier_name": null,
  "client_name": null,
  "invoice_number": null,
  "invoice_date": null,
  "due_date": null,
  "amount_ht": null,
  "vat_amount": null,
  "total_amount": null,
  "currency": "XAF|EUR|USD",
  "payment_status": "paid|unpaid",
  "confidence": 0.0,
  "description": null
}}

Texte :
{text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt]
    )

    raw = response.text

    try:
        clean = clean_json(raw)
        data = json.loads(clean)

        # 🔥 sécuriser montants
        data["amount_ht"] = parse_amount(data.get("amount_ht"))
        data["vat_amount"] = parse_amount(data.get("vat_amount"))
        data["total_amount"] = parse_amount(data.get("total_amount"))

        return data

    except Exception:
        logger.warning("❌ JSON parsing failed")
        return {"raw_response": raw}


# =====================================================
# API
# =====================================================

@app.post("/analyze-document")
async def analyze_document(
    id_document: int = Form(...),
    document_url: str = Form(...)
):
    try:
        logger.info(f"📄 Document {id_document}")

        # 1. Télécharger image
        resp = requests.get(document_url, timeout=20)
        resp.raise_for_status()

        image_bytes = resp.content

        # 2. OCR
        text = extract_text(image_bytes)

        if not text:
            return {
                "success": False,
                "id_document": id_document,
                "error": "Aucun texte OCR"
            }

        # 3. Gemini
        result = analyze_with_gemini(text)

        return {
            "success": True,
            "id_document": id_document,
            "documents": [result]
        }

    except Exception as e:
        logger.exception("❌ Erreur API")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================
# RUN
# =====================================================

""" if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) """