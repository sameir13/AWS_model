import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import os
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__root__)


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

file_path = 'dataset.csv'
texts, labels = load_data(file_path)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenize_data(train_texts, tokenizer)
val_encodings = tokenize_data(val_texts, tokenizer)

def convert_to_tf_dataset(encodings, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset

train_dataset = convert_to_tf_dataset(train_encodings, train_labels)
val_dataset = convert_to_tf_dataset(val_encodings, val_labels)

train_dataset = train_dataset.shuffle(len(train_dataset)).batch(16)
val_dataset = train_dataset.batch(16)

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

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

try:
    loss, accuracy = model.evaluate(val_dataset)
    logger.info(f'Validation Loss: {loss}')
    logger.info(f'Validation Accuracy: {accuracy}')
except Exception as e:
    logger.error(f"An error occurred during model evaluation: {str(e)}")
    raise

def detect_ai_content(text, model, tokenizer):
    """Detects the likelihood that a given text is AI-generated."""
    try:
        inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        ai_generated_prob = probs[0][1].numpy()
        return ai_generated_prob * 100  
    except Exception as e:
        logger.error(f"An error occurred during AI content detection: {str(e)}")
        raise


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



if __name__ == '__main__':
    app.run(debug=True)
