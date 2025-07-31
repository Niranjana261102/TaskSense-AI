import pandas as pd
from src.data_processing import DataProcessor
from config import Config
import os
import numpy as np

def generate_sample_data(n_samples=1000):
    categories = Config.CATEGORIES
    descriptions = []
    labels = []
    
    components = ['login', 'authentication', 'database', 'UI', 'API', 'payment', 'user management']
    
    for _ in range(n_samples):
        category = np.random.choice(categories)
        component = np.random.choice(components)
        
        if category == 'Bug':
            description = f"Fix {component} system crash"
        elif category == 'Feature':
            description = f"Add new {component} functionality"
        elif category == 'Testing':
            description = f"Create tests for {component} module"
        elif category == 'Documentation':
            description = f"Update {component} documentation"
        else:  # DevOps
            description = f"Setup {component} deployment pipeline"
            
        descriptions.append(description)
        labels.append(category)
    
    df = pd.DataFrame({
        'description': descriptions,
        'category': labels
    })
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate and save data
    df = generate_sample_data()
    df.to_csv('data/raw/task_data.csv', index=False)
    print(f"Generated {len(df)} sample tasks")
    print("\nSample data:")
    print(df.head())
    print("\nCategory distribution:")
    print(df['category'].value_counts())
