from datasets import load_dataset
from Config import *
import torch
from transformers import DistilBertTokenizer
from transformers import AutoModel
from EmotionDataset import EmotionDataset
from torch.utils.data import DataLoader, random_split

def tokenize(tokenizer, batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_states(model, tokenizer, batch):
    inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

def load_datasets():
    dataset = load_dataset(DATA_SET)

    model_checkpoint = "distilbert-base-uncased"

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)

    emotions_encoded = dataset.map(lambda x: tokenize(distilbert_tokenizer, x), batched=True, batch_size=None)

    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)

    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    emotions_hidden = emotions_encoded.map(lambda x: extract_hidden_states(model, distilbert_tokenizer, x), batched=True)

    X_train = emotions_hidden["train"]["hidden_state"]
    y_train = emotions_hidden["train"]["label"]

    X_val = emotions_hidden["validation"]["hidden_state"]
    y_val = emotions_hidden["validation"]["label"]

    X_test = emotions_hidden["test"]["hidden_state"]
    y_test = emotions_hidden["test"]["label"]

    train_data = EmotionDataset(X_train, y_train)
    val_data = EmotionDataset(X_val, y_val)
    test_data = EmotionDataset(X_test, y_test)

    partition_size_train = len(train_data) // NUM_CLIENTS
    lengths_train = [partition_size_train] * NUM_CLIENTS

    partition_size_val = len(val_data) // NUM_CLIENTS
    lengths_val = [partition_size_val] * NUM_CLIENTS
    
    train_datasets = random_split(train_data, lengths_train, torch.Generator().manual_seed(42))
    val_datasets = random_split(val_data, lengths_val, torch.Generator().manual_seed(42))

    train_loaders = []
    val_loaders = []

    for ds in train_datasets:
        train_loaders.append(DataLoader(ds, batch_size=BATCH_SIZE))
    
    for ds in val_datasets:
        val_loaders.append(DataLoader(ds, batch_size=BATCH_SIZE))

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    print(len(train_loaders))

    return train_loaders, val_loaders, test_loader,