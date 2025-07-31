import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from config import Config

class DataProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        # Fit the label encoder with known categories immediately
        self.label_encoder.fit(Config.CATEGORIES)
        
    def prepare_data(self, data_path):
        # Read data
        df = pd.read_csv(data_path)
        
        # Tokenize texts
        encoded_texts = self.tokenizer(
            df['description'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Convert labels to numerical format
        labels = pd.Series(self.label_encoder.transform(df['category']))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_texts['input_ids'],
            labels,
            test_size=0.2,
            random_state=42
        )
        
        return X_train, X_test, y_train, y_test, self.label_encoder.classes_