# DA5401_Assignment4_ED22B054: GMM-Based Synthetic Sampling for Imbalanced Data

**Objective:**  
This project applies a **Gaussian Mixture Model (GMM)** to generate synthetic samples for the minority class in a highly imbalanced credit card fraud dataset and evaluates its impact on classifier performance compared to a baseline Logistic Regression model.

**Dataset:**  
Credit Card Fraud Detection Dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 1. Problem Statement

The dataset contains a large imbalance between **non-fraudulent** and **fraudulent transactions**. A baseline classifier trained on the imbalanced data tends to **ignore the minority class**. The project implements a **GMM-based oversampling pipeline** and a **CBU+GMM balancing approach** to improve the model's detection of fraudulent transactions.

---

## 2. Baseline Model

- Trained a **Logistic Regression classifier** on the original imbalanced training data.
- Evaluated using **Precision, Recall, F1-score**, and **ROC-AUC**.
- Results for the **minority class (fraudulent transactions)**:

| Metric      | Value  |
|------------|--------|
| Precision  | 0.861  |
| Recall     | 0.628  |
| F1-score   | 0.727  |
| ROC-AUC    | 0.964  |

> The model has high precision but **misses a significant number of minority samples**, highlighting the need for oversampling.

---

## 3. GMM-Based Synthetic Sampling

- **Fitted a Gaussian Mixture Model** on the minority class of the training data.
- Determined the **optimal number of components (k=7)** using **BIC**.
- Generated synthetic minority samples to balance the training set.
- Combined with original data for training a new Logistic Regression model.

**Selected GMM Details (scaled space):**

- **Weights (`pi_k`)**: `[0.0262, 0.093, 0.375, 0.061, 0.1395, 0.0116, 0.2936]`
- **Means**: see notebook for full 30-feature vectors.

---

## 4. CBU + GMM Balancing

- Applied **Clustering-Based Undersampling (CBU)** to reduce majority class.
- Used **GMM-based synthetic sampling** on minority class to match majority population.
- Resulted in a **balanced training dataset** for model evaluation.

---

## 5. Model Performance After Oversampling

**Minority class metrics comparison:**

| Model               | Precision | Recall | F1-score |
|--------------------|----------|--------|----------|
| Baseline            | 0.861    | 0.628  | 0.727    |
| GMM-Oversampled     | 0.077    | 0.858  | 0.141    |
| CBU+GMM Balanced    | 0.075    | 0.858  | 0.138    |

**Observations:**

- **Recall increased significantly** for both GMM-based models, indicating more minority samples were correctly detected.
- **Precision dropped**, resulting in more false positives.
- **F1-score decreased**, but in imbalanced classification, **recall for the minority class is often more critical**.

---

## 6. Comparative Visualization

- Created a **bar chart comparing Precision, Recall, and F1-score** for the three models.
- Shows the trade-off between **high recall** and **low precision** for GMM-based oversampling.

---

## 7. Final Recommendation

- **GMM-based oversampling is effective** for improving detection of minority class samples.
- The method captures **complex substructures** in the minority distribution better than simple oversampling techniques like SMOTE.
- **CBU+GMM** gives similar recall to GMM-only oversampling but may provide better diversity in the majority class.
- For use cases where **recall is critical** (e.g., fraud detection), GMM-based synthetic sampling is recommended.  
- **Further improvement** can be achieved by threshold tuning, cost-sensitive learning, or ensemble methods to mitigate the precision drop.

---

*This project demonstrates how probabilistic modeling with GMM can help address class imbalance and improve minority class detection in critical applications like fraud detection.*
