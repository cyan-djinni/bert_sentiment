# 🎬 BERT Sentiment Analysis on IMDB

**Languages:** English | [中文](#chinese) | [Deutsch](#german) | [Français](#french)

---

## 📌 Overview

This project fine-tunes the pre-trained transformer model `bert-base-uncased` for **binary sentiment classification** on the IMDB movie review dataset.

The implementation covers a complete NLP pipeline:

- Data preprocessing and cleaning  
- Transformer fine-tuning  
- Validation and test evaluation  
- Interactive inference  

The final model was trained on the **full dataset** using **GPU acceleration in Google Colab**.

---

## 🗂 Dataset

**IMDB Movie Review Dataset** (Kaggle version)

Binary labels:

- `0` — Negative  
- `1` — Positive  

Reviews are cleaned and tokenized using the BERT tokenizer with a maximum sequence length of 128.

---

## 🧠 Methodology

- **Base Model:** `bert-base-uncased`  
- **Framework:** PyTorch + HuggingFace Transformers  
- **Max Sequence Length:** 128  
- **Optimizer:** AdamW  
- **Evaluation Metrics:** Accuracy, Weighted F1-score  

A linear classification head is fine-tuned on top of the pre-trained BERT encoder.

---

## ⚙️ Environment Setup

```bash
conda create -n bert_env python=3.10
conda activate bert_env
pip install -r requirements.txt


⸻

🔄 Data Preprocessing

python src/preprocess.py

This step performs dataset cleaning and prepares the data for model training.

⸻

🚀 Training

Local development training:

python src/train.py

Full-scale training was conducted in Google Colab (GPU) on the complete dataset.

Trained model weights are not included in this repository due to size constraints.

⸻

📊 Evaluation

python src/evaluate.py --model_path PATH_TO_CHECKPOINT

Final Model Performance (Full Dataset, GPU Training)

Metric	Validation	Test
Accuracy	92.64%	89.28%
Weighted F1	92.67%	89.28%

The slight decrease from validation to test performance indicates mild overfitting while maintaining strong generalization ability.

⸻

🔍 Inference

python src/predict.py

Example:

Enter a review: This movie was fantastic.
Prediction: positive


⸻

📁 Project Structure

bert_sentiment/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md


⸻



中文说明

本项目基于预训练模型 bert-base-uncased 对 IMDB 电影评论进行二分类情感分析。

完整流程包括：数据清洗、模型微调、验证与测试评估、交互式预测。

最终模型使用完整数据集，并在 Google Colab GPU 环境下训练。

最终结果：
	•	Validation F1: 92.67%
	•	Test F1: 89.28%

验证集与测试集之间存在轻微性能差异，属于正常泛化现象。

⸻



Deutsche Version

Dieses Projekt fine-tuned das vortrainierte Modell bert-base-uncased für eine binäre Sentimentanalyse auf dem IMDB-Datensatz.

Der vollständige Workflow umfasst Datenvorverarbeitung, Training, Evaluation und Inferenz.

Das finale Modell wurde mit dem kompletten Datensatz unter GPU-Beschleunigung in Google Colab trainiert.

Ergebnisse:
	•	Validation F1: 92.67%
	•	Test F1: 89.28%

Die leichte Differenz zwischen Validierungs- und Testleistung deutet auf mildes Overfitting hin.

⸻



Version Française

Ce projet entraîne le modèle pré-entraîné bert-base-uncased pour une classification binaire des sentiments sur le dataset IMDB.

Le pipeline comprend le prétraitement des données, l’entraînement, l’évaluation et l’inférence.

Le modèle final a été entraîné sur l’ensemble complet des données avec accélération GPU sur Google Colab.

Résultats finaux :
	•	Validation F1 : 92.67%
	•	Test F1 : 89.28%

La légère baisse entre validation et test indique un surapprentissage modéré avec une bonne capacité de généralisation.