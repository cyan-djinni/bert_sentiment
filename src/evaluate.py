from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1️⃣ 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2️⃣ 加载 test 数据
test_df = pd.read_csv("/content/bert_sentiment/data/Test_clean.csv")
test_dataset = Dataset.from_pandas(test_df)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 3️⃣ 加载 checkpoint 模型（本地）
model = BertForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/bert_run1/checkpoint-5000",
    local_files_only=True
)

# 4️⃣ 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# 5️⃣ 创建 Trainer（只用于 evaluate）
training_args = TrainingArguments(
    output_dir="./temp",
    per_device_eval_batch_size=64,   # CPU 上建议调大
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

# 6️⃣ 评估
results = trainer.evaluate(test_dataset)
print(results)