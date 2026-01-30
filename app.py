from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import os

from prediction import load_model, predict  # same as Streamlit

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load model once at startup
model = load_model()

# Define categories
categories = [
    'Central Serous Chorioretinopathy_Color Fundus',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

@app.route('/predict', methods=['POST'])
def classify_retina():
    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Read image and convert to RGB
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Run prediction
        probabilities = predict(model, image)  # NumPy array
        max_index = int(np.argmax(probabilities))

        return jsonify({
            'prediction': categories[max_index],
            'confidence': float(probabilities[max_index]),
            'probabilities': {
                categories[i]: float(probabilities[i])
                for i in range(len(categories))
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# IMPORTANT: Render needs this
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
