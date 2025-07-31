import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from src.model import TaskClassifier  # Updated import path
import numpy as np
from sklearn.metrics import classification_report
import logging
from tqdm import tqdm

def train_model(X_train, y_train, X_test, y_test, num_classes, epochs=5):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, torch.tensor(y_train.values))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = TaskClassifier(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'models/task_classifier.pth')
    logger.info("Model saved successfully")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_dataset = TensorDataset(X_test, torch.tensor(y_test.values))
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        for batch in test_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            outputs = model(input_ids, attention_mask=None)
            predictions = torch.argmax(outputs, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print classification report
    report = classification_report(all_labels, all_preds)
    logger.info(f"\nClassification Report:\n{report}")
    
    return model