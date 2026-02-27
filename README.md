
# 🎬 BERT Sentiment Analysis on IMDB

Languages: English | [中文](#chinese) | [Deutsch](#german) | [Français](#french)

## Overview

This project fine-tunes the pre-trained transformer model `bert-base-uncased` for binary sentiment classification on the IMDB movie review dataset.

The objective is to construct a complete NLP workflow including:

- Data preprocessing  
- Transformer fine-tuning  
- Model evaluation  
- Interactive inference  

The current version represents a local development experiment prior to full-scale GPU training.

---

## Dataset

IMDB Movie Review Dataset (Kaggle version)

Binary sentiment labels:

- 0 — Negative  
- 1 — Positive  

---

## Method

Base model: `bert-base-uncased`  
Framework: PyTorch + HuggingFace Transformers  

Fine-tuning is performed using supervised learning with a linear classification head on top of the BERT encoder.

---

## Environment

```bash
conda create -n bert_env python=3.10
conda activate bert_env
pip install -r requirements.txt
````

---

## Training

```bash
python src/train.py
```

The fine-tuned model is saved to:

```
outputs/bert_finetuned
```

---

## Inference

```bash
python src/predict.py
```

Example:

```
Enter a review: This movie was fantastic.
Prediction: positive
```

---

## Results

Validation Accuracy: 0.83

Training configuration:

* CPU training
* Reduced dataset (development setting)

Full training with GPU acceleration will be conducted in a subsequent phase using Google Colab.

---

## Project Structure

```
bert_sentiment/
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── outputs/
├── requirements.txt
└── README.md
```

---

<a id="chinese"></a>

## 中文说明

本项目基于预训练模型 `bert-base-uncased` 对 IMDB 电影评论进行二分类情感分析。

当前版本为本地开发阶段实验结果（小规模数据 + CPU 训练）。
后续将在 GPU 环境中进行完整训练与性能分析。

---

<a id="german"></a>

## Deutsche Version

Dieses Projekt fine-tuned das vortrainierte Modell `bert-base-uncased` für eine binäre Sentimentanalyse auf dem IMDB-Datensatz.

Die aktuelle Version basiert auf einem lokalen Entwicklungsdurchlauf mit reduzierten Daten.
Eine vollständige GPU-Trainingsphase ist geplant.

---

<a id="french"></a>

## Version Française

Ce projet entraîne le modèle pré-entraîné `bert-base-uncased` pour une classification binaire des sentiments sur le dataset IMDB.

La version actuelle correspond à une phase de développement local avec un sous-ensemble réduit de données.
Un entraînement complet avec GPU sera réalisé ultérieurement.


