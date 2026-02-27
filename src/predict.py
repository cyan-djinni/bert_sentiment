import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../outputs/bert_finetuned"
label_map = {0: "negative", 1: "positive"}

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    return label_map[pred]

if __name__ == "__main__":
    tokenizer, model = load_model()

    while True:
        text = input("Enter a review (or type 'quit'): ")
        if text.lower() == "quit":
            break

        result = predict(text, tokenizer, model)
        print("Prediction:", result)