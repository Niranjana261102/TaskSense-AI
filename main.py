import argparse
import logging
from src.data_processing import DataProcessor
from src.train import train_model
from src.predict import TaskPredictor
from src.model import TaskClassifier
from config import Config
import torch
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Task Classification System')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--text', help='Text to classify (for predict mode)')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Ensure required directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    if args.mode == 'train':
        logger.info("Starting training process...")
        
        # Initialize data processor
        processor = DataProcessor()
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, classes = processor.prepare_data(Config.DATA_PATH)
            
            # Train model
            model = train_model(
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                len(Config.CATEGORIES),
                Config.EPOCHS
            )
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    elif args.mode == 'predict':
        if not args.text:
            logger.error("Text argument is required for predict mode")
            return
            
        logger.info("Starting prediction process...")
        
        try:
            # Load model and create predictor
            model = TaskClassifier(len(Config.CATEGORIES))
            model.load_state_dict(torch.load(Config.MODEL_PATH))
            processor = DataProcessor()
            predictor = TaskPredictor(model, processor.label_encoder)
            
            # Make prediction
            predicted_category = predictor.predict(args.text)
            logger.info(f"Input text: {args.text}")
            logger.info(f"Predicted category: {predicted_category}")
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

if __name__ == "__main__":
    main()