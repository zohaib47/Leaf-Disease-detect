import io
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- Model Setup from Hugging Face ---
MODEL_NAME = "zohaib47/potato-leaf-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device).eval()
labels = model.config.id2label

# --- Google Gemini API Setup ---
# Render ke environment variables se API key uthayega
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-3.5-flash-lite")

LOW_CONF_THRESHOLD = 70.0  # ViT ki confidence (%) isse kam ho to warning flag

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {"status": "ok", "device": str(device), "classes": list(labels.values())}
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image nahi mili"}), 400
    try: 
        image_file = request.files["image"]
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Ye valid image file nahi hai"}), 400

    # Step 1: Google Gemini API se check karwayein ke yeh kis sabzi ka patta hai
    try:
        response = gemini_model.generate_content([
            (
                "Look at this image. Is this a plant leaf? If yes, which vegetable"
                " or plant leaf is this? Reply strictly in this format: 'Plant Name"
                " | Yes' if it is a leaf, or 'Not a leaf | No' if it is not a leaf."
            ),
            img,
        ])
        result_text = response.text.strip()

        # Parsing Gemini response
        if "Not a leaf" in result_text or "No" in result_text:
            return (
                jsonify({
                    "error": (
                        "Ye leaf ki image nahi lagti. Please leaf ki saaf image"
                        " upload karein."
                    ),
                    "not_leaf": True,
                }),
                422,
            )

        # Plant ka naam nikalna
        plant_name = (
            result_text.split("|")[0].strip()
            if "|" in result_text
            else result_text
        )

    except Exception as e:
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

    # Step 2: Aapka trained model disease prediction karega
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    results = sorted(
        [
            {"label": labels[i], "probability": round(p * 100, 2)}
            for i, p in enumerate(probs)
        ],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return jsonify({
        "plant_detected": plant_name,
        "prediction": results[0]["label"],
        "confidence": results[0]["probability"],
        "low_confidence": results[0]["probability"] < LOW_CONF_THRESHOLD,
        "all": results,
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Image 10 MB se bari hai"}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)