# 💬 Sentiment Analysis Dashboard Using Machine Learning in Python

This project focuses on **Sentiment Analysis using Machine Learning in Python**. The notebook implements a complete NLP pipeline to classify text into **positive**, **negative**, or **neutral** sentiments using classical ML models.

📍 **Notebook Link**: [Open in Google Colab](https://colab.research.google.com/drive/1H1Ie4nqeTlU009TI6FVJtGemT61XtFhz)

---

## 🔍 Overview

The objective of this task is to analyze textual data (e.g., tweets or product reviews) and build machine learning models that accurately predict the sentiment behind the text.

### ✅ Key Features:

* Data cleaning and preprocessing using **NLTK**
* Word tokenization, stopword removal, and lemmatization
* Visualization with **Matplotlib** and **Seaborn**
* Feature extraction using **TF-IDF Vectorizer**
* Classification using:

  * Logistic Regression
  * Naive Bayes
  * Support Vector Machine (SVM)
* Model evaluation with:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report
* Feature importance analysis for interpretability
* Error analysis of misclassified examples

---

## 🛠️ Technologies Used

| Tool                 | Purpose                        |
| -------------------- | ------------------------------ |
| Python               | Programming Language           |
| Pandas               | Data Handling                  |
| NLTK                 | Natural Language Preprocessing |
| Scikit-learn         | ML Models & Evaluation         |
| Matplotlib / Seaborn | Data Visualization             |
| WordCloud            | Word Frequency Visualization   |
| Google Colab         | Cloud Notebook Environment     |

---

## 📂 Dataset

The notebook works with a `tweets_sentiment.csv` file, consisting of:

* `text`: Raw text data (tweets or reviews)
* `sentiment`: Sentiment label (positive, negative, neutral)

> 🔁 If the dataset is not available, a small sample dataset is loaded automatically for testing.

---

## 🧹 Text Preprocessing Steps

1. Lowercasing
2. URL and special character removal
3. Tokenization using `nltk.word_tokenize()`
4. Stopword removal
5. Lemmatization using `WordNetLemmatizer`
6. Final cleaned text column created for modeling

---

## 🧠 Models Implemented

| Model                   | Description                                   |
| ----------------------- | --------------------------------------------- |
| Logistic Regression     | Best performing model with interpretability   |
| Multinomial Naive Bayes | Fast and probabilistic                        |
| Support Vector Machine  | Accurate and robust for high-dimensional data |

All models are trained using **TF-IDF vectorized text** and evaluated on a test set using multiple metrics.

---

## 📊 Results

* ✅ Best Accuracy: **Logistic Regression**
* 🔍 Feature Importance:

  * Positive → “love”, “excellent”, “amazing”
  * Negative → “worst”, “terrible”, “bad”

---

## 📌 Recommendations

* Add more labeled data to improve model generalization
* Apply advanced NLP models like **BERT** for better contextual understanding
* Use **Stratified Splits** and **class balancing** for fair model training
* Incorporate emoji/emoticon analysis for richer sentiment signals

---

## 📈 Screenshots & Plots

* Sentiment distribution plot
* Confusion matrices for each model
* Accuracy comparison bar chart
* Top feature words per sentiment
* Sample misclassified texts

---

## 📚 How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1H1Ie4nqeTlU009TI6FVJtGemT61XtFhz)
2. Upload or use the sample dataset
3. Run each cell in order
4. Review output, visualizations, and model evaluation results

---

## 🙋‍♂️ Author

**Name**: *TANMAY GUHA*

**Email**: tanmayguha15@gmail.com

---

## 📌 License

This project is open for educational and personal use. Please credit if reused.

---

