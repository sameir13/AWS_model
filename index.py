import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import os
from flask import Flask, request, jsonify

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load and Prepare the Data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("The dataset must contain 'text' and 'label' columns.")
        texts = data['text'].tolist()
        labels = data['label'].tolist()
        logger.info("Data loaded successfully.")
        return texts, labels
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing the CSV file.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {str(e)}")
        raise

# Replace 'dataset.csv' with the path to your dataset file
file_path = 'dataset.csv'
texts, labels = load_data(file_path)

# Split the data into training and validation sets (80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 2: Tokenize the Data
def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenize_data(train_texts, tokenizer)
val_encodings = tokenize_data(val_texts, tokenizer)

# Step 3: Convert to TensorFlow Datasets
def convert_to_tf_dataset(encodings, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset

train_dataset = convert_to_tf_dataset(train_encodings, train_labels)
val_dataset = convert_to_tf_dataset(val_encodings, val_labels)

# Batch and shuffle the datasets
train_dataset = train_dataset.shuffle(len(train_dataset)).batch(16)
val_dataset = train_dataset.batch(16)

# Step 4: Fine-Tune BERT Model
def build_and_compile_model():
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    return model

model = build_and_compile_model()

try:
    model.fit(train_dataset, epochs=3, validation_data=val_dataset)
    logger.info("Model training completed successfully.")
except Exception as e:
    logger.error(f"An error occurred during model training: {str(e)}")
    raise

# Save the model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Step 5: Evaluate the Model
try:
    loss, accuracy = model.evaluate(val_dataset)
    logger.info(f'Validation Loss: {loss}')
    logger.info(f'Validation Accuracy: {accuracy}')
except Exception as e:
    logger.error(f"An error occurred during model evaluation: {str(e)}")
    raise

# Step 6: Define a function for AI content detection
def detect_ai_content(text, model, tokenizer):
    """Detects the likelihood that a given text is AI-generated."""
    try:
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        ai_generated_prob = probs[0][1].numpy()
        return ai_generated_prob * 100  # Convert to percentage
    except Exception as e:
        logger.error(f"An error occurred during AI content detection: {str(e)}")
        raise

# Function to load the model for future inference
def load_model(model_path='./fine_tuned_model'):
    try:
        model = TFBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        logger.info("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {str(e)}")
        raise

model, tokenizer = load_model()

# API endpoint for AI content detection
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        paragraph = data.get('paragraph', None)
        
        if not paragraph:
            return jsonify({'error': 'No paragraph provided.'}), 400
        
        ai_percentage = detect_ai_content(paragraph, model, tokenizer)
        
        if ai_percentage > = 60:
            result = "Text is AI-generated."
        else:
            result = "Text is human-written."
        
        response = {
            'ai_percentage': ai_percentage,
            'result': result
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An error occurred in the /detect endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred during detection.'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
