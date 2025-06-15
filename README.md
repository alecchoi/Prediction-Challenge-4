🚗 Car Deal Quality Predictor

This project builds a predictive model to classify car deals as **Good**, **Neutral**, or **Poor** using machine learning techniques in R. It utilizes a **random forest (ranger)** classifier with **5-fold cross-validation** and **upsampling** to improve generalization from a small training set to a much larger testing set.

Achieved **84% accuracy** on a test set of 9,800 entries.

---

## 📁 Project Structure

```
car-deal-quality-predictor/
│
├── Prediction4Train.csv               # Training data (200 rows)
├── Prediction4StudentsTest.csv       # Test data for prediction (9,800 rows)
├── submission.csv                    # Output predictions: ID and Deal
├── car_deal_quality_predictor.R     # Main script (training + inference)
└── README.md                         # This file
```

---

## 🧠 Model Details

* **Model**: Random Forest (via `ranger`)
* **Framework**: `caret`
* **Validation**: 5-Fold Cross-Validation
* **Sampling Strategy**: Upsampling to address class imbalance
* **Final Accuracy**: \~84% on held-out test set
* **Prediction Target**: `Deal` (`Good`, `Neutral`, `Poor`)

---

## 📦 Requirements

Install the following R packages:

```r
install.packages("caret")
install.packages("ranger")
```

---

## 🚀 How to Run

1. Clone the repository and set your R working directory accordingly.
2. Ensure both `Prediction4Train.csv` and `Prediction4StudentsTest.csv` are present.
3. Run the script:

```r
source("car_deal_quality_predictor.R")
```

This will:

* Train the model with 5-fold CV + upsampling.
* Evaluate it using a confusion matrix and print accuracy.
* Predict labels for the large student test set.
* Output `submission.csv`.

---

## 📈 Sample Output

```
Accuracy: 84.12 %

Confusion Matrix:
           Reference
Prediction  Good Neutral Poor
    Good     ...    ...   ...
    Neutral  ...    ...   ...
    Poor     ...    ...   ...
```

---

## 📤 Submission File Format

| ID | Deal    |
| -- | ------- |
| 1  | Neutral |
| 2  | Poor    |
| 3  | Good    |
| …  | …       |

---
