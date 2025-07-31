class Config:
    # Model parameters
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    
    # Data parameters
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Categories
    CATEGORIES = ['Bug', 'Feature', 'Testing', 'Documentation', 'DevOps']
    
    # Paths
    MODEL_PATH = 'models/task_classifier.pth'
    DATA_PATH = 'data/raw/task_data.csv'
    PROCESSED_DATA_PATH = 'data/processed/'