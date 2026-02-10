from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import os
import pandas as pd
import uuid

# ===== Flask App =====
BASE_DIR = r"D:\Nails Project"
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ===== Paths =====
MODEL_PATH = os.path.join(BASE_DIR, "Models", "Vgg-16-nail-disease.h5")
CSV_PATH = os.path.join(BASE_DIR, "Data", "nail_diseases_dataset.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== Allowed file extensions =====
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== Load trained model =====
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ===== Load CSV file =====
try:
    df = pd.read_csv(CSV_PATH)

    required_cols = ["Disease_Name", "Symptoms", "Cause", "Severity", "Treatment"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    print("✅ CSV loaded successfully")

except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    df = pd.DataFrame(columns=["Disease_Name", "Symptoms", "Cause", "Severity", "Treatment"])

# ===== Class names =====
class_names = [
    "alopecia_areata", "yellow_nails", "beau_s_lines", "bluish_nail", "clubbing",
    "darier_s_disease", "eczema_nail", "lindsay_s_nails", "koilonychia", "leukonychia",
    "muehrcke_s_lines", "onycholysis", "pale_nail", "red_lunula", "splinter_hemorrhage",
    "terry_s_nail", "white_nail"
]

# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ✅ Nail page (Upload + Result both here)
@app.route("/nail")
def nail():
    return render_template("nailpred.html", prediction=None)

# ✅ Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "❌ Model not loaded. Check server logs."

    if "file" not in request.files:
        return redirect(url_for("nail"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("nail"))

    if not allowed_file(file.filename):
        return "❌ Only PNG, JPG, JPEG files are allowed."

    # ===== Save uploaded file =====
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ===== Load and preprocess image =====
    # Change this:
    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===== Predict =====
    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions))
    confidence = round(float(np.max(predictions) * 100), 2)
    result = class_names[predicted_class_index]

    # ===== Match prediction with CSV =====
    search_name = result.replace("_", " ").lower()
    disease_info = df[df["Disease_Name"].str.lower() == search_name]

    if not disease_info.empty:
        info_dict = disease_info.iloc[0].to_dict()
    else:
        info_dict = {
            "Disease_Name": result,
            "Symptoms": "No data found",
            "Cause": "No data found",
            "Severity": "No data found",
            "Treatment": "No data found"
        }

    image_url = url_for('static', filename=f"uploads/{filename}")

    return render_template(
        "nailpred.html",
        prediction=result,
        confidence=confidence,
        image_path=image_url,
        info=info_dict
    )

# ===== Run server =====
if __name__ == "__main__":
    app.run(debug=True)