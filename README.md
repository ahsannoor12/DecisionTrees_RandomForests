# 🌳 Decision Trees & Random Forests

A hands-on machine learning lab exploring tree-based classification models — from single Decision Trees to powerful ensemble methods.

---

## 📌 About This Lab

**Course:** Applied Machine Learning  
**Week:** 5 | **Lab:** 3 | **Session:** 23  
**Topic:** Classification Models II — Building Powerful Ensemble Models  
**Dataset:** Wine Dataset & Breast Cancer Dataset (from `sklearn`)

---

## 📂 Files in This Repo

| File | Description |
|------|-------------|
| `Week5_Lab3_DecisionTrees_RandomForests.ipynb` | Main lab notebook |
| `README.md` | Project documentation |

---

## 🎯 What This Lab Covers

| Part | Topic |
|------|-------|
| Part 1 | Setup, Data Loading & Exploration |
| Part 2 | Building & Visualizing Decision Trees |
| Part 3 | Random Forests & Ensemble Learning |
| Part 4 | Feature Importance Analysis |
| Part 5 | Hyperparameter Tuning with GridSearchCV |
| Part 6 | Model Comparison & Feature Selection |

---

## 🧠 Concepts Learned

- ✅ How Decision Trees split data using **Gini Impurity**
- ✅ Why deep trees **overfit** and how to control it with `max_depth`
- ✅ How **Random Forests** reduce overfitting using Bootstrap Sampling + Voting
- ✅ How to rank features using **Feature Importance**
- ✅ How to find best settings using **GridSearchCV**
- ✅ Difference between **Bagging** (Random Forest) vs **Boosting** (Gradient Boost / XGBoost)

---

## 📊 Models Compared

| Model | Training Accuracy | Test Accuracy |
|-------|:-----------------:|:-------------:|
| Decision Tree (Full) | 100% | ~94% |
| Decision Tree (Depth=3) | ~96% | ~94% |
| Random Forest (Default) | 100% | ~97% |
| Random Forest (Optimized) | 100% | ~97–100% |
| Gradient Boosting | ~100% | ~97% |
| XGBoost | ~100% | ~97–100% |

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier  # optional
```

---

## 🚀 How to Run

1. Clone this repo:
```bash
git clone https://github.com/ahsannoor12/DecisionTrees_RandomForests.git
```

2. Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

3. Open the notebook:
```bash
jupyter notebook Week5_Lab3_DecisionTrees_RandomForests.ipynb
```

> No dataset downloads needed — all data loads automatically from `sklearn.datasets`

---

## 💡 Key Takeaway

> Using only the **Top 5 most important features** (out of 13) gave nearly the same accuracy as using all features — proving that simpler models can be just as powerful!

---

*Made with ❤️ | ahsannoor12*
