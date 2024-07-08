import pandas as pd
import logging
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import os
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_and_unify_datasets(file_path1, file_path2):
    try:
        data1 = pd.read_csv(file_path1)
        data2 = pd.read_csv(file_path2)
        
        if 'text' not in data1.columns or 'label' not in data1.columns:
            raise ValueError(f"The dataset {file_path1} must contain 'text' and 'label' columns.")
        if 'text' not in data2.columns or 'label' not in data2.columns:
            raise ValueError(f"The dataset {file_path2} must contain 'text' and 'label' columns.")
        
        label_mapping = {
            'positive': 1, 'good': 1, 
            'negative': 0, 'bad': 0, 
            'neutral': 2, 'average': 2
        }
        
        data1['unified_label'] = data1['label'].map(label_mapping)
        data2['unified_label'] = data2['label'].map(label_mapping)
        
        combined_data = pd.concat([data1[['text', 'unified_label']], data2[['text', 'unified_label']]])
        combined_data.columns = ['text', 'label']
        
        texts = combined_data['text'].tolist()
        labels = combined_data['label'].tolist()
        
        logger.info("Datasets loaded and unified successfully.")
        return texts, labels
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the CSV file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the datasets: {str(e)}")
        raise

file_path1 = 'ai_reviews.csv'
file_path2 = 'movie_reviews.csv'
texts, labels = load_and_unify_datasets(file_path1, file_path2)

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
val_dataset = val_dataset.batch(16)

def build_and_compile_model():
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    return model

model = build_and_compile_model()

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        logger.info(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

try:
    history = model.fit(train_dataset, epochs=3, validation_data=val_dataset, callbacks=[CustomCallback()])
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
        
        if ai_percentage >= 60:
            result = "Text is AI-generated."
        else:
            result = "Text is human-written."
        
        response = {
            'ai_percentage': ai_percentage,
            'result': result,
            'train_loss': history.history['loss'],
            'train_accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'],
            'epochs': len(history.history['loss'])
        }
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An error occurred in the /detect endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred during detection.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
