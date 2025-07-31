from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import TaskPredictor
from src.model import TaskClassifier
from src.data_processing import DataProcessor
import torch
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize model and predictor
model = TaskClassifier(len(Config.CATEGORIES))
model.load_state_dict(torch.load(Config.MODEL_PATH))
processor = DataProcessor()
predictor = TaskPredictor(model, processor.label_encoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        prediction = predictor.predict(text)
        return jsonify({
            'text': text,
            'category': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)