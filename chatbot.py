import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

class Chatbot:
    def __init__(self, model_path='model', label_map_path='label_to_answer.json', dataset_path='train.csv'):
        # Load fine-tuned classification model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Load label to answer map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)

        # Load training questions
        import pandas as pd
        df = pd.read_csv(dataset_path)
        self.questions = df['question'].tolist()

        # Load model and encode training questions
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.question_embeddings = self.embedder.encode(self.questions, convert_to_tensor=True)

    def get_response(self, user_input):
        #Semantic Search
        user_embedding = self.embedder.encode(user_input, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(user_embedding, self.question_embeddings)[0]
        top_idx = torch.argmax(similarity_scores).item()
        matched_question = self.questions[top_idx]

        #Classification
        inputs = self.tokenizer(matched_question, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        return self.label_map.get(str(predicted_label), "I'm not sure how to answer that.")

#CLI interface
if __name__ == "__main__":
    bot = Chatbot()
    print("AI Safety Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break
        response = bot.get_response(query)
        print("Bot:", response)

