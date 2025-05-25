#To train DistilBERT model for fine tuning chatbot responses and save weights

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

#load and split dataset
df = pd.read_csv('train.csv')
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['question'], df['label'], test_size=0.2, random_state=42)

#tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

#
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = QADataset(train_encodings, list(train_labels))
val_dataset = QADataset(val_encodings, list(val_labels))

#load model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(df['label'].unique()))

#training args
training_args = TrainingArguments(
    output_dir='./model',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    warmup_steps=20,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
    gradient_accumulation_steps=2,
    weight_decay=0.01
)


#trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

#train
trainer.train()

#save model
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
print("Model saved to ./model")
