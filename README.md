# TaskSense AI: Intelligent Task Classification System

**TaskSense AI** is an intelligent task classification system built with deep learning and natural language processing (NLP). It automatically classifies task descriptions into predefined categories like Bug, Feature, Testing, Documentation, and DevOps using a fine-tuned BERT model.

---

## 🚀 Features

- ✅ Classifies task descriptions into 5 categories
- 🤖 Fine-tuned BERT model for high-accuracy classification
- 🌐 Simple web interface using Flask and HTML/CSS
- 📊 Evaluation metrics with confusion matrix visualization
- 🧪 Model training with logging and metrics
- 📝 Sample data generation for quick experimentation

---

## 🧠 Categories

- Bug
- Feature
- Testing
- Documentation
- DevOps

---

## 🛠️ Technologies Used

| Area             | Tools/Frameworks                              |
|------------------|------------------------------------------------|
| Frontend         | HTML, CSS, JavaScript (Vanilla)               |
| Backend          | Python, Flask                                 |
| Machine Learning | PyTorch, Transformers (HuggingFace BERT)      |
| Data             | pandas, scikit-learn                          |
| Visualization    | seaborn, matplotlib                           |
| Utilities        | logging, argparse                             |

---

## ⚙️ How It Works

1. **Data Generation**:
   - Uses `generate.py` to create sample task data with categories and descriptions.
2. **Data Preprocessing**:
   - Tokenizes text using BERT tokenizer.
   - Converts categories to numerical labels.
3. **Model Training**:
   - Trains a custom BERT-based classifier using `train.py`.
   - Saves the trained model to `models/task_classifier.pth`.
4. **Evaluation**:
   - Confusion matrix and classification report for performance analysis.
5. **Prediction**:
   - Load model and predict category using `predict.py` or the Flask web interface.
6. **Web App**:
   - `app.py` serves a simple form to enter task description and shows category result dynamically.

---

🎥 Watch the project in action: 

<video src="path-to-video.mp4" controls></video>

