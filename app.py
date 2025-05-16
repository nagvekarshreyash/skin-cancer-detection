from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize models
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov8m/best.pt')
yolo_model = YOLO(YOLO_MODEL_PATH)

# Configure Gemini AI
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash-lite')

def preprocess_image(image):
    # Resize image to model's required size
    img = cv2.resize(image, (256, 256))
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def is_skin_image(image):
    try:
        pil_image = Image.fromarray(image)
        prompt = """Analyze this image and determine if it shows human skin or a skin condition. 
        If it shows skin or a skin condition, respond with 'yes'. 
        If it shows anything else, respond with 'no'."""
        response = model_gemini.generate_content([prompt, pil_image])
        return "yes" in response.text.lower()
    except Exception as e:
        print(f"Error in is_skin_image: {str(e)}")
        return False

def check_malignancy(image):
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Prepare image for ViT model
        inputs = vit_processor(images=pil_image, return_tensors="pt")
        
        # Get prediction
        with torch.no_grad():
            outputs = vit_model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
            
        # Get predicted class and confidence
        predicted_class = torch.argmax(predictions).item()
        confidence = predictions[0][predicted_class].item()
        
        # Assuming class 1 is malignant (adjust based on your model's classes)
        is_malignant = predicted_class == 1
        
        return is_malignant, confidence
    except Exception as e:
        print(f"Error in check_malignancy: {str(e)}")
        return False, 0.0

def get_precautions(class_name, confidence):
    try:
        prompt = f"""Provide 3 brief, essential precautions for {class_name} skin condition:
        1. Basic care
        2. Warning signs
        3. Prevention
        Keep it very short and simple."""
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting precautions: {str(e)}")
        return "Unable to generate precautions at this time."

def check_suspicious_patterns(image):
    try:
        pil_image = Image.fromarray(image)
        prompt = """Analyze this skin image for potential cancer patterns. Look for:
        1. Irregular or asymmetrical spots
        2. Uneven borders or edges
        3. Multiple colors or dark spots
        4. Diameter larger than 6mm
        5. Any changes or evolution
        
        If ANY of these patterns are present, respond with 'suspicious'.
        If the skin appears normal without concerning patterns, respond with 'normal'.
        Be very specific in your response."""
        response = model_gemini.generate_content([prompt, pil_image])
        return 'suspicious' in response.text.lower()
    except Exception as e:
        print(f"Error in check_suspicious_patterns: {str(e)}")
        return True  # Default to suspicious if error occurs

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file in request'
            }), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No selected file'
            }), 400
        
        # Process image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'status': 'error',
                'message': 'Failed to decode image'
            }), 400
            
        processed_image = preprocess_image(image)
        
        # Check if it's a skin image
        if not is_skin_image(processed_image):
            return jsonify({
                'status': 'error',
                'message': 'The uploaded image is not a skin image.'
            }), 400
        
        # Step 2: Check for suspicious patterns using Gemini
        has_suspicious_patterns = check_suspicious_patterns(processed_image)
        
        # If no suspicious patterns, return benign
        if not has_suspicious_patterns:
            return jsonify({
                'status': 'success',
                'condition': 'benign',
                'confidence': 0.95,
                'yolo_predictions': [{
                    'class': 'normal_skin',
                    'confidence': 0.95,
                    'precaution': 'Continue regular skin checks and sun protection.'
                }]
            })
        
        # Step 3: If suspicious patterns found, run YOLO model
        results = yolo_model(processed_image, verbose=False)
        
        response_data = {
            'status': 'success',
            'condition': 'malignant',
            'confidence': 0.92,
            'yolo_predictions': []
        }
        
        if results and len(results) > 0:
            predictions = []
            for r in results:
                if hasattr(r, 'probs'):
                    probs = r.probs
                    class_id = int(probs.top1)
                    confidence = float(probs.top1conf)
                    class_name = yolo_model.names[class_id] if class_id in yolo_model.names else f"class_{class_id}"
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': round(confidence, 2),
                        'precaution': get_precautions(class_name)
                    })
            
            response_data['yolo_predictions'] = predictions
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def get_precautions(condition_type):
    precautions = {
        'melanoma': 'Seek immediate medical attention. Protect from sun exposure. Monitor for changes.',
        'basal_cell': 'Consult dermatologist. Avoid sun exposure. Use sunscreen regularly.',
        'squamous_cell': 'Seek medical evaluation. Protect from UV rays. Regular skin checks.',
        # Add other conditions as needed
    }
    return precautions.get(condition_type, 'Please consult a healthcare professional for proper evaluation.')

# At the top of your file, add:
from werkzeug.middleware.proxy_fix import ProxyFix

# After creating the Flask app, add:
app.wsgi_app = ProxyFix(app.wsgi_app)

# Update the CORS configuration
CORS(app, resources={
    "*": {"origins": "*"}
})

# In your main block, update:
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200