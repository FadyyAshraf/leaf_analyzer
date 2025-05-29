import os
import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from flask import Flask, request, jsonify

app = Flask(__name__)

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
LEAF_MODEL_PATH         = os.path.join(BASE_DIR, "leafNetV3_model.tflite")
DISEASE_MODEL_PATH      = os.path.join(BASE_DIR, "converted_model.tflite")      # MobileNetV3
MOBILEVIT_MODEL_PT_PATH = os.path.join(BASE_DIR, "mobilevit_model17.pt")          # MobileViT (PyTorch)

# === Load Models ===
leaf_model = tf.lite.Interpreter(model_path=LEAF_MODEL_PATH)
leaf_model.allocate_tensors()

disease_model_v3 = tf.lite.Interpreter(model_path=DISEASE_MODEL_PATH)
disease_model_v3.allocate_tensors()

# Load PyTorch MobileViT model
mobilevit_model = torch.load(MOBILEVIT_MODEL_PT_PATH, map_location=torch.device('cpu'))
mobilevit_model.eval()

# === MobileViT Preprocessing ===
# Normalization values used during MobileViT training
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]

mobilevit_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# === Other Preprocessing Functions ===
def preprocess_image_mobilenet(image_bytes, target_size):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def preprocess_image(image_bytes, input_shape):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    input_data = np.expand_dims(img_array, axis=0)
    return input_data

def preprocess_image_mobilevit_pt(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = mobilevit_transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# === Inference ===
def run_tflite(interpreter, input_data):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])[0]

def run_pytorch_model(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        return output[0].numpy()

# === Main Function ===
def analyze_leaf_disease(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # --- 1) Leaf Detection ---
    leaf_shape = leaf_model.get_input_details()[0]["shape"]
    leaf_input = preprocess_image_mobilenet(image_bytes, (leaf_shape[2], leaf_shape[1]))
    leaf_output = run_tflite(leaf_model, leaf_input)

    leaf_class = int(np.argmax(leaf_output))
    leaf_confidence = float(leaf_output[leaf_class])

    if leaf_class != 0:
        return json.dumps({
            "success":     True,
            "stage":       "leaf_detection",
            "isLeaf":      False,
            "confidence":  leaf_confidence,
            "message":     "This is not a leaf",
            "results":     []
        })

    # --- 2) Disease Detection from both models ---
    disease_labels = [
        "Apple: Apple scab", "Apple: Black rot", "Apple: Cedar apple rust",
        "Apple: healthy", "Grape: Esca (Black Measles)", "Pepper-bell: Bacterial-spot",
        "Pepper-bell: healthy", "Potato: Early blight", "Potato: Late blight",
        "Potato: healthy", "Strawberry: Leaf scorch", "Strawberry: healthy",
        "Tomato: Bacterial spot", "Tomato: Early blight", "Tomato: Late blight",
        "Tomato: Leaf Mold", "Tomato: healthy"
    ]

    # --- MobileNetV3 Inference (TFLite) ---
    shape_v3 = disease_model_v3.get_input_details()[0]['shape']
    input_v3 = preprocess_image(image_bytes, shape_v3)
    output_v3 = run_tflite(disease_model_v3, input_v3)

    # --- MobileViT Inference (PyTorch) ---
    input_vit_tensor = preprocess_image_mobilevit_pt(image_bytes)
    output_vit = run_pytorch_model(mobilevit_model, input_vit_tensor)

    # --- Pick best confidence ---
    class_v3 = int(np.argmax(output_v3))
    class_vit = int(np.argmax(output_vit))
    conf_v3 = float(output_v3[class_v3])
    conf_vit = float(output_vit[class_vit])

    if conf_v3 >= conf_vit:
        final_label = disease_labels[class_v3]
        final_conf = conf_v3
    else:
        final_label = disease_labels[class_vit]
        final_conf = conf_vit

    result = {
        "success":         True,
        "stage":           "disease_detection",
        "isLeaf":          True,
        "leaf_confidence": leaf_confidence,
        "results":         [{"label": final_label, "confidence": final_conf}],
        "message":         "Analysis complete"
    }

    return json.dumps(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
