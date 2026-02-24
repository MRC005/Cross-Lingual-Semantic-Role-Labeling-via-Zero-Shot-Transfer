from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-multilingual-cased"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModel.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")