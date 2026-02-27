# src/train.py

import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import os

# --------------------
# 配置
# --------------------
DATA_DIR = "data"
MODEL_NAME = "bert-base-uncased"  # 如果你想用 cased 可以改成 bert-base-cased
BATCH_SIZE = 4                  # Colab 免费 GPU 适合小 batch
EPOCHS = 2                        # 第一次跑用 2 epoch 快速测试
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# 读取数据
# --------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
valid_df = pd.read_csv(os.path.join(DATA_DIR, "valid_clean.csv"))

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
train_dataset = train_dataset.select(range(1000))
val_dataset = valid_dataset.select(range(400))
# --------------------
# Tokenizer
# --------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)

# Hugging Face 只需要 label 列名叫 "labels"
train_dataset = train_dataset.rename_column("label", "labels")
valid_dataset = valid_dataset.rename_column("label", "labels")

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# --------------------
# Metrics
# --------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# --------------------
# 模型
# --------------------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --------------------
# TrainingArguments
# --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    dataloader_pin_memory=False,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

# --------------------
# 开始训练
# --------------------
trainer.train()

# --------------------
# 保存模型
# --------------------
trainer.save_model(os.path.join(OUTPUT_DIR, "bert_finetuned"))
print("训练完成，模型保存在 outputs/bert_finetuned")