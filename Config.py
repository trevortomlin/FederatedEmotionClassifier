from pathlib import Path
import torch

NUM_CLIENTS = 10
BATCH_SIZE = 32
DATA_SET = "emotion"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_MODEL_PATH = Path("distilbert-base-uncased-finetuned-emotion/checkpoint-500")