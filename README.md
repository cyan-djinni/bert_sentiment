⸻

🎬 BERT Sentiment Analysis on IMDB

Languages: English | 中文 | Deutsch | Français

⸻

Overview

This project fine-tunes the pre-trained transformer model bert-base-uncased for binary sentiment classification on the IMDB movie review dataset.

The project implements a complete NLP pipeline including:
	•	Data preprocessing
	•	Transformer fine-tuning
	•	Model evaluation on validation and test sets
	•	Interactive inference

The final model was trained on the full cleaned dataset using GPU acceleration in Google Colab.

⸻

Dataset

IMDB Movie Review Dataset (Kaggle version)

Binary sentiment labels:
	•	0 — Negative
	•	1 — Positive

The dataset was cleaned and tokenized prior to training.

⸻

Method

Base model: bert-base-uncased
Framework: PyTorch + HuggingFace Transformers

Fine-tuning is performed using supervised learning with a linear classification head on top of the BERT encoder.

Maximum sequence length: 128
Optimization: AdamW
Evaluation metrics: Accuracy and weighted F1-score

⸻

Environment

conda create -n bert_env python=3.10
conda activate bert_env
pip install -r requirements.txt


⸻

Training

Local development training:

python src/train.py

Full-scale training was conducted in Google Colab using GPU acceleration.

The trained model checkpoint is stored locally and is not included in this repository due to size constraints.

⸻

Evaluation

python src/evaluate.py --model_path PATH_TO_CHECKPOINT


⸻

Final Results

Final model (full dataset, Colab GPU training):
	•	Validation F1: 92.67%
	•	Validation Accuracy: 92.64%
	•	Test F1: 89.28%
	•	Test Accuracy: 89.28%

The slight decrease from validation to test performance indicates mild overfitting but overall strong generalization capability.

⸻

Inference

python src/predict.py

Example:

Enter a review: This movie was fantastic.
Prediction: positive


⸻

Project Structure

bert_sentiment/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md


⸻



中文说明

本项目基于预训练模型 bert-base-uncased 对 IMDB 电影评论进行二分类情感分析。

最终模型使用完整数据集，并在 Google Colab GPU 环境中进行训练。

最终结果：
	•	Validation F1: 92.67%
	•	Test F1: 89.28%

验证集与测试集之间存在轻微性能下降，属于正常泛化差异，整体模型表现稳定。

⸻



Deutsche Version

Dieses Projekt fine-tuned das vortrainierte Modell bert-base-uncased für eine binäre Sentimentanalyse auf dem IMDB-Datensatz.

Das finale Modell wurde mit dem vollständigen Datensatz unter Verwendung von GPU-Beschleunigung in Google Colab trainiert.

Ergebnisse:
	•	Validation F1: 92.67%
	•	Test F1: 89.28%

Die leichte Leistungsdifferenz zwischen Validierungs- und Testdaten deutet auf ein mildes Overfitting hin, die Generalisierungsfähigkeit bleibt jedoch stabil.

⸻



Version Française

Ce projet entraîne le modèle pré-entraîné bert-base-uncased pour une classification binaire des sentiments sur le dataset IMDB.

Le modèle final a été entraîné sur l’ensemble complet des données avec accélération GPU sur Google Colab.

Résultats finaux :
	•	Validation F1 : 92.67%
	•	Test F1 : 89.28%

La légère baisse entre validation et test indique un surapprentissage modéré mais une bonne capacité de généralisation.

⸻
