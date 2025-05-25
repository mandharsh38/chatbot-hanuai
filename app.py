#flask app
from flask import Flask, request, jsonify
import torch
import json
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load models and data
MODEL_PATH = 'model'
LABEL_MAP_PATH = 'label_to_answer.json'
DATASET_PATH = 'train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

df = pd.read_csv(DATASET_PATH)
questions = df['question'].tolist()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

# Define API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({"error": "No query provided"}), 400

        # Step 1: Find most similar question
        user_embedding = embedder.encode(user_input, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
        top_idx = torch.argmax(similarity_scores).item()
        matched_question = questions[top_idx]

        # Step 2: Classify matched question
        inputs = tokenizer(matched_question, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        answer = label_map.get(str(predicted_label), "I'm not sure how to answer that.")
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
