import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments
from Config import *
from LoadDataset import load_datasets
from transformers import DistilBertTokenizer
from transformers import AutoModel
from LoadDataset import tokenize, extract_hidden_states

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

def train_local_encoder():
    dataset = load_datasets(DATA_SET)
    model_checkpoint = "distilbert-base-uncased"
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)
    emotions_encoded = dataset.map(lambda x: tokenize(distilbert_tokenizer, x), batched=True, batch_size=None)
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    emotions_hidden = emotions_encoded.map(lambda x: extract_hidden_states(model, distilbert_tokenizer, x), batched=True)

    batch_size = 64
    logging_steps = len(emotions_encoded["train"]) // batch_size
    model_name = f"{model_checkpoint}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    log_level="error",
                                    )

    trainer = Trainer(model=model,
                        args=training_args,
                        compute_metrics=compute_metrics,
                        train_dataset=emotions_encoded["train"],
                        eval_dataset=emotions_encoded["validation"],
                        tokenizer=distilbert_tokenizer)

    trainer.train()