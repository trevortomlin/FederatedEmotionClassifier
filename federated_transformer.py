import itertools
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
from transformers import DistilBertTokenizer
from transformers import AutoModel
import numpy as np
from transformers import AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

DATA_SET = "emotion"

dataset = load_dataset(DATA_SET)

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = dataset.map(tokenize, batched=True, batch_size=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-emotion/checkpoint-500").to(device)

def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

X_train = emotions_hidden["train"]["hidden_state"]
y_train = emotions_hidden["train"]["label"]

X_val = emotions_hidden["validation"]["hidden_state"]
y_val = emotions_hidden["validation"]["label"]

X_test = emotions_hidden["test"]["hidden_state"]
y_test = emotions_hidden["test"]["label"]

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
score = log_reg.score(X_val, y_val)
print(f"Logistic Regression: {score}")


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_val, y_val)
print(f"Dummy Classifier: {score}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(768, 6)
        

    def forward(self, x):
        x = self.fc1(x)
        return x

mymodel = NeuralNetwork()

learning_rate = 0.1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate)

num_epochs = 2
for epoch in range(num_epochs):
    correct = 0
    for i, (X, y) in enumerate(zip(X_train, y_train)):
        y_pred = mymodel(X)

        loss = loss_fn(y_pred, y)

        correct += (torch.argmax(y_pred) == y).float().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 4000 == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(X_train)}], Loss: {loss.item():.4f}")
        #     accuracy = correct / (i+1) * 100
        #     print(f"Accuracy: {accuracy:.2f}")

print("Training Complete")

with torch.no_grad():
    correct = 0
    for i, (X, y) in enumerate(zip(X_val, y_val)):

        correct += (torch.argmax(mymodel(X)) == y).float().item()

    accuracy = correct / len(X_val) * 100
    print(f"Accuracy: {accuracy:.2f}")

# def plot_confusion_matrix(y_preds, y_true, labels):
#     cm = confusion_matrix(y_true, y_preds, normalize="true")
#     fig, ax = plt.subplots(figsize=(6, 6))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format=".2f")
#     plt.title("Normalized Confusion Matrix")
#     plt.show()

#y_preds = log_reg.predict(X_val)
#plot_confusion_matrix(y_preds, y_val, emotions_encoded["train"].features["label"].names)

# num_labels = 6
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)

# from sklearn.metrics import accuracy_score, f1_score

# def compute_metrics(p):
#     preds, labels = p
#     preds = np.argmax(preds, axis=1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds, average="weighted"),
#     }

# from transformers import Trainer, TrainingArguments

# batch_size = 64
# logging_steps = len(emotions_encoded["train"]) // batch_size
# model_name = f"{model_checkpoint}-finetuned-emotion"
# training_args = TrainingArguments(output_dir=model_name,
#                                   num_train_epochs=2,
#                                   learning_rate=2e-5,
#                                   per_device_train_batch_size=batch_size,
#                                   per_device_eval_batch_size=batch_size,
#                                   weight_decay=0.01,
#                                   evaluation_strategy="epoch",
#                                   disable_tqdm=False,
#                                   logging_steps=logging_steps,
#                                   log_level="error",
#                                 )

# trainer = Trainer(model=model,
#                     args=training_args,
#                     compute_metrics=compute_metrics,
#                     train_dataset=emotions_encoded["train"],
#                     eval_dataset=emotions_encoded["validation"],
#                     tokenizer=tokenizer)

# trainer.train()