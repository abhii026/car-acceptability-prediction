# 🚗 Car Acceptability Prediction

> A simple, interpretable machine learning project that predicts whether a car is **unacceptable, acceptable, good, or very good** from six attributes. It compares a Decision Tree, KNN, Logistic Regression, and Naive Bayes on the UCI Car Evaluation dataset.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| ML Models | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) (Decision Tree, KNN, Logistic Regression, Naive Bayes) |
| Statistics | SciPy (chi-square test, corrected t-test) |
| Visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) |
| Environment | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) |

---

## 📁 Folder Structure

```
car-acceptability-prediction/
├── car_acceptability.ipynb     ← Main notebook (run top to bottom)
├── analysis_extra.py           ← Extra analyses: 20-split test, depth and k sensitivity
├── data/
│   ├── car.data                ← Raw dataset (1,728 cars, no header row)
│   └── car.names               ← Dataset description (UCI)
├── figures/                    ← Plots used in the paper and saved by the notebook
├── results/                    ← Result tables (CSV)
├── research_paper/
│   └── car_acceptability_paper.pdf
├── requirements.txt            ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

1. Load the UCI Car Evaluation dataset and check for missing values and duplicates.
2. Look at the class distribution. About 70% of cars are `unacc`, so the data is imbalanced.
3. Encode the six attributes as ordered integers (for example, `low < med < high < vhigh`).
4. Split the data 80/20 into training (1,382 cars) and test (346 cars) sets, stratified by class.
5. Choose the decision tree depth with 5-fold cross-validation on the training set.
6. Train five models: majority baseline, Logistic Regression, Decision Tree, KNN (k=5), and Naive Bayes.
7. Evaluate with accuracy and macro F1, then check the differences with repeated cross-validation and a corrected t-test.
8. Check stability across 20 different splits, and analyse feature importance and decision rules.

---

## 📊 Model Results (Test Set, 346 cars)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Decision Tree (depth 6)** | **0.939** | **0.812** | **0.870** | **0.838** |
| KNN (k = 5) | 0.916 | 0.925 | 0.766 | 0.816 |
| Naive Bayes | 0.841 | 0.671 | 0.510 | 0.542 |
| Logistic Regression | 0.801 | 0.669 | 0.557 | 0.598 |
| Majority baseline | 0.699 | 0.175 | 0.250 | 0.206 |

**Confusion matrix: Decision Tree**

| Actual \ Predicted | unacc | acc | good | vgood |
|---|---|---|---|---|
| unacc | 235 | 7 | 0 | 0 |
| acc | 1 | 68 | 6 | 2 |
| good | 0 | 0 | 12 | 2 |
| vgood | 0 | 3 | 0 | 10 |

**Key findings**
- The Decision Tree gives the best result on the main split. Across 20 different splits, KNN has a slightly higher average macro F1 (0.865 vs 0.833). The two are statistically tied in repeated cross-validation (p = 0.35).
- Logistic Regression and Naive Bayes are clearly worse, because the car rules depend on combinations of attributes.
- **Most influential features:** `safety` and `persons`. Number of doors has almost no effect.

![Feature Importance](figures/fig_importance.pdf)
![Test-set Results](figures/fig_test.pdf)

---

## 🧬 Concepts Used

- **Ordinal encoding** of ordered attributes
- **Stratified train/test split** to keep class proportions
- **Class imbalance handling** through macro F1, per-class recall, and confusion matrices
- **Cross-validation and corrected paired t-test** (Nadeau and Bengio) for model comparison
- **Feature importance:** Cramér's V, decision tree impurity, and permutation importance
- **Interpretability:** the final decision tree is printed as plain if-then rules

---

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_ID/car-acceptability-prediction.git
cd car-acceptability-prediction
```

### 2. Install the requirements
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook car_acceptability.ipynb
```
Run all cells from top to bottom (**Cell → Run All**). It takes a few seconds. Run it from the project folder, because the paths are relative.

### 4. (Optional) Run the extra analyses
```bash
python analysis_extra.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
scikit-learn
jupyter
notebook
```

---

## 📚 Dataset

UCI Car Evaluation dataset (Bohanec, 1997).
Source: https://doi.org/10.24432/C5JP48

---

## 👨‍💻 Author

- **Abhishek Singh** (12406001) — [GitHub](https://github.com/YOUR_GITHUB_ID) · iabhii026@gmail.com

Lovely Professional University, Department of Computer Science and Engineering

---

## 🙏 Thank You

If you found this project useful, consider giving it a ⭐ on GitHub.
