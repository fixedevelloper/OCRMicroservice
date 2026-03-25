from fastapi import FastAPI, Form, HTTPException
import requests
import json
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
import pytesseract
import cv2
import numpy as np
from google import genai
import re
from pdf2image import convert_from_bytes

# =====================================================
# CONFIG LOGGING
# =====================================================

LOG_FILE = "/var/www/accountIa/storage/logs/ocr_api.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("ocr_api")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =====================================================
# CONFIG APP
# =====================================================

load_dotenv()

# 🔥 Forcer chemin Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI(title="OCR + Gemini API")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY manquante")

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================================
# OCR
# =====================================================

def preprocess_image(image_bytes: bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("❌ Image decode failed")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # debug image
        cv2.imwrite("/tmp/debug.jpg", gray)

        return gray

    except Exception:
        logger.exception("❌ preprocess_image error")
        return None


def extract_images_from_pdf(pdf_bytes: bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        return images  # liste d’images PIL
    except Exception:
        logger.exception("❌ PDF conversion error")
        return []
    
def extract_text(image_bytes: bytes, content_type: str):
    try:
        texts = []

        # ===== CAS PDF =====
        if "pdf" in content_type:
            logger.info("📄 PDF détecté")

            images = extract_images_from_pdf(image_bytes)

            for i, img_pil in enumerate(images):
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)

                text = pytesseract.image_to_string(
                    img,
                    lang="fra+eng",
                    config="--oem 3 --psm 4"
                )

                if text.strip():
                    texts.append(text)

        # ===== CAS IMAGE =====
        else:
            img = preprocess_image(image_bytes)

            text = pytesseract.image_to_string(
                img,
                lang="fra+eng",
                config="--oem 3 --psm 4"
            )

            if not text.strip():
                logger.warning("⚠️ fallback RAW image")
                nparr = np.frombuffer(image_bytes, np.uint8)
                raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                text = pytesseract.image_to_string(
                    raw_img,
                    lang="fra+eng",
                    config="--oem 3 --psm 4"
                )

            texts.append(text)

        final_text = "\n".join(texts)

        logger.info(f"OCR total length: {len(final_text)}")

        return final_text.strip()

    except Exception:
        logger.exception("❌ OCR global error")
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
    try:
        logger.info("🤖 Analyse Gemini lancée")

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

        clean = clean_json(raw)
        data = json.loads(clean)

        data["amount_ht"] = parse_amount(data.get("amount_ht"))
        data["vat_amount"] = parse_amount(data.get("vat_amount"))
        data["total_amount"] = parse_amount(data.get("total_amount"))

        return data

    except Exception:
        logger.exception("❌ Gemini parsing error")
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
        logger.info(f"📄 Document ID: {id_document}")
        logger.info(f"🌐 URL: {document_url}")

        # Télécharger image
        resp = requests.get(document_url, timeout=20)
        resp.raise_for_status()

        image_bytes = resp.content
        logger.info(f"📦 Image size: {len(image_bytes)} bytes")
        content_type = resp.headers.get("Content-Type", "")
        # OCR
        text = extract_text(image_bytes, content_type)

        if not text:
            logger.error("❌ Aucun texte OCR détecté")
            return {
                "success": False,
                "id_document": id_document,
                "error": "Aucun texte OCR"
            }

        # Gemini
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