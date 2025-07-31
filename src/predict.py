import torch
from transformers import BertTokenizer
from src.model import TaskClassifier  # Updated import path

class TaskPredictor:
    def __init__(self, model, label_encoder):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = label_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def predict(self, text):
        # Prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=None)
            predictions = torch.argmax(outputs, dim=1)
            
        # Convert to label
        predicted_label = self.label_encoder.inverse_transform(predictions.cpu().numpy())[0]
        return predicted_label