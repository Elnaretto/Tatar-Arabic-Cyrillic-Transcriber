#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:06:24 2026

@author: elnarbaynazarov
"""

# app.py
from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import pytesseract
from normalize import clean_text, normalize_arabic
from transliterate import transliterate, latin_to_cyrillic
import io

app = Flask(__name__)

# -----------------------------
# 1. Preprocess image
# -----------------------------
def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    gray = ImageOps.grayscale(image)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)

    # Resize
    gray = gray.resize((gray.width*2, gray.height*2))

    # Numpy for OpenCV
    img_np = np.array(gray)

    # Median filter + threshold
    img_np = cv2.medianBlur(img_np, 3)
    _, img_np = cv2.threshold(img_np, 140, 255, cv2.THRESH_BINARY)

    return Image.fromarray(img_np)

# -----------------------------
# 2. OCR + normalization + transliteration
# -----------------------------
def ocr_transcribe(file_bytes):
    processed = preprocess_image(file_bytes)
    ocr_text = pytesseract.image_to_string(processed, lang="ara", config="--psm 6")
    clean = clean_text(ocr_text)
    norm = normalize_arabic(clean)
    latin = transliterate(norm)
    cyrillic = latin_to_cyrillic(latin)
    return {
        "ocr": ocr_text,
        "latin": latin,
        "cyrillic": cyrillic
    }

# -----------------------------
# 3. Flask route
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error":"No file"}), 400
    file = request.files["image"]
    result = ocr_transcribe(file.read())
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
