# To convert dataset.json to train.csv with labels for training

import json
import pandas as pd

with open('dataset.json', 'r') as f:
    data = json.load(f)

questions = []
labels = []
answers = []

for idx, pair in enumerate(data):
    questions.append(pair['question'])
    labels.append(idx)  # Assign index as label
    answers.append(pair['answer'])

df = pd.DataFrame({'question': questions, 'label': labels})
df.to_csv('train.csv', index=False)

label_map = {idx: ans for idx, ans in enumerate(answers)}
with open('label_to_answer.json', 'w') as f:
    json.dump(label_map, f, indent=2)

print("done")
