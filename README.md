````markdown
# 🎓 GATE Question Topic Classifier using DistilBERT

This repository contains a fine-tuned **DistilBERT** model for **topic classification of GATE questions**. It uses a labeled dataset of GATE-style questions across multiple subjects and predicts the corresponding topic label.

---

## 🚀 Features

- 🔍 Text classification using **DistilBERT**
- 🧠 Trained on GATE question dataset with labeled topics
- 📊 Evaluation with **Accuracy** and **F1 Score**
- 🧪 Inference-ready model with easy prediction interface
- 💾 Model, Tokenizer, and Label Encoder persistence supported

---

## 🧾 Dataset

The dataset used is a CSV file with the following columns:

- `question` — The GATE question text
- `topic` — The manually annotated topic label

---

## 📦 Requirements

Install the required packages:

```bash
pip install transformers tensorflow scikit-learn pandas
````

---

## 🛠️ Training Pipeline

### 1. Preprocessing

* Load CSV data
* Drop rows with missing values
* Encode topic labels using `LabelEncoder`
* Split into train (70%), validation (15%), and test (15%) sets

### 2. Tokenization

Tokenized using `DistilBertTokenizer` with:

* `max_length = 128`
* Padding and truncation

### 3. Model

* Model: `TFDistilBertForSequenceClassification`
* Optimizer: Adam with `learning_rate=2e-5`
* Loss: Sparse Categorical Crossentropy
* Metrics: Sparse Categorical Accuracy

### 4. Training

```python
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)
```

---

## 📈 Evaluation

After training, the model is evaluated on the test set:

```python
Test Accuracy: 0.9241
Test F1 Score: 0.9182
```

(*Example results, may vary*)

---

## 🔍 Inference

Use the `predict()` function to classify new GATE questions:

```python
predict([
    "Consider a demand paging memory management system with 32-bit logical address..."
])
```

Returns:

```
['Operating Systems']
```

---

## 💾 Saving the Model

The model, tokenizer, and label encoder are saved to disk:

```python
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))
```

---

## 📁 Project Structure

```
.
├── questions-topic-classification.ipynb
├── questions-data-new.csv
├── main_model/
│   ├── config.json
│   ├── tf_model.h5
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── label_encoder.pkl
```

