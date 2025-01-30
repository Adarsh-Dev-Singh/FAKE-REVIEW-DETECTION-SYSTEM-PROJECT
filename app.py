from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch

app = Flask(__name__)
CORS(app)  


model_path = "./model_checkpoints/checkpoint-2757"  
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

df_kaggle = pd.read_csv("amazon_reviews.txt", sep="\t")

# Convert labels to readable format
label_mapping = {'__label2__': 'fake', '__label1__': 'real'}
df_kaggle['LABEL'] = df_kaggle['LABEL'].map(label_mapping)

# Combine title and text into a single field
df_kaggle['TEXT'] = df_kaggle.apply(
    lambda x: f"{x['REVIEW_TITLE']}. {x['REVIEW_TEXT']}", axis=1
)

# Route to get unique product categories
@app.route('/categories', methods=['GET'])
def get_categories():
    categories = df_kaggle['PRODUCT_CATEGORY'].unique().tolist()
    return jsonify({"categories": categories})

# Updated route to send all parameters
@app.route('/reviews', methods=['POST'])
def get_reviews():
    data = request.json
    category = data.get("category")
    if not category:
        return jsonify({"error": "No category provided"}), 400

    # Filter reviews by the selected category
    category_reviews = df_kaggle[df_kaggle['PRODUCT_CATEGORY'] == category]
    
    # Select a random sample of 10 reviews (or fewer if not enough reviews exist)
    random_reviews = category_reviews.sample(n=10, random_state=42) if len(category_reviews) >= 10 else category_reviews

    # Generate predictions
    def predict_text(text):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        return "fake" if prediction == 0 else "real"

    random_reviews['MODEL_ANALYSIS'] = random_reviews['TEXT'].apply(predict_text)

    # Convert to dictionary format with all fields
    reviews = random_reviews.to_dict(orient='records')
    return jsonify({"reviews": reviews})

# Route to predict if a review is fake or real
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    true_label = data.get("true_label", "")  # Optionally receive the true label for validation
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    sentiment = "fake" if prediction == 0 else "real"

    return jsonify({
        "prediction": sentiment,
        "true_label": true_label,
        "is_correct": sentiment == true_label  # Validate prediction if true_label is provided
    })

if __name__ == '__main__':
    app.run(debug=True)
