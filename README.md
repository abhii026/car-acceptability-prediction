# Car Acceptability Prediction

**Author:** Abhishek Singh (12406001)
**Email:** iabhii026@gmail.com
**GitHub:** [@YOUR_GITHUB_ID](https://github.com/YOUR_GITHUB_ID)

A simple machine learning project that predicts how acceptable a car is
(`unacc`, `acc`, `good`, `vgood`) from six attributes: buying price, maintenance cost,
number of doors, passenger capacity, luggage boot size, and safety.

The project compares **five simple models** and explains the best one with plain if-then rules.
The full write-up is in `research_paper/car_acceptability_paper.pdf`.

## Results

Test set: 346 cars (20% of the data, stratified).

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Decision Tree (depth 6)** | **0.939** | **0.838** |
| KNN (k = 5) | 0.916 | 0.816 |
| Naive Bayes | 0.841 | 0.542 |
| Logistic Regression | 0.801 | 0.598 |
| Majority-class baseline | 0.699 | 0.206 |

Most cars are `unacc` (70%), so accuracy alone is misleading. Macro F1 is the main metric.

Across 20 different train/test splits, KNN has a slightly higher average macro F1 than the
Decision Tree (0.865 vs 0.833). The two models are statistically tied in repeated cross-validation.

**Most important features:** `persons` and `safety`. Number of doors has almost no effect.

## Dataset

UCI Car Evaluation dataset (Bohanec, 1997). Source: <https://doi.org/10.24432/C5JP48>

- 1,728 cars, 6 features, no missing values, no duplicates
- Each row is a unique combination of the six features, and labels come from a fixed expert rule model

The raw file is `data/car.data`. The column names are in `data/car.names`.

## Project structure

```
12406001_37_AP/
├── car_acceptability.ipynb   # main analysis notebook (run from top to bottom)
├── analysis_extra.py         # extra analyses used in the paper (20 splits, depth, k)
├── data/
│   ├── car.data              # raw dataset (no header row)
│   └── car.names             # dataset description
├── figures/                  # figures used in the paper and saved by the notebook
├── results/                  # result tables (CSV)
├── research_paper/
│   └── car_acceptability_paper.pdf
├── requirements.txt          # Python packages
├── .gitignore
└── README.md
```

## How to run

Run everything from the project folder, because the paths are relative.

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
2. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Open the notebook:**
   ```bash
   jupyter notebook car_acceptability.ipynb
   ```
4. **Run all cells** (Cell > Run All). It takes a few seconds.
5. **(Optional) Run the extra analyses:**
   ```bash
   python analysis_extra.py
   ```

## What the notebook does

| Section | Content |
|---|---|
| 1 | Load the data and check for missing values and duplicates |
| 2 | Look at the class distribution (imbalance) |
| 3 | Encode the features as ordered numbers |
| 4 | Split into 80% training and 20% test |
| 5 | Choose the decision tree depth with 5-fold cross-validation, then build the 5 models |
| 6 | Train and test every model |
| 7 | Compare the models (chart and table) |
| 8 | Confusion matrix and per-class report for the best model |
| 9 | Feature importance from the decision tree |
| 10 | Predict the acceptability of a new car |
| 11 | Conclusion |

## Models

| Model | Why it is included |
|---|---|
| Majority-class baseline | Reference point: always predicts the most common class |
| Logistic Regression | Simple linear model |
| Decision Tree | Easy to read as if-then rules (best model) |
| K-Nearest Neighbors (k=5) | Predicts by looking at the most similar cars |
| Naive Bayes | Probabilistic model that assumes independent features |

## Notes

- Ordinal encoding is used because the levels have a natural order (`low < med < high`).
- The decision tree depth is limited to 6 so the tree stays readable.
- Results use one fixed random seed (`random_state=42`) for the main split.
  The 20-split analysis in `analysis_extra.py` shows how much the scores vary between splits.

## License and citation

The dataset is public (UCI Machine Learning Repository). If you use this project, please cite the paper in `research_paper/`.
