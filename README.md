#  Comment Category Prediction

Built an end-to-end **machine learning pipeline** to classify user-generated comments into **platform-defined categories** using text, metadata, and engagement features.

---

##  Problem Statement
The objective of this project is to predict the final category assigned to a comment based on:

- Text content
- User engagement signals
- Metadata
- Platform-generated indicators

This is a **multi-class text classification problem** with **4 target classes**.

---

##  Dataset
The dataset contains:

- `train.csv` → input features + target (`label`)
- `test.csv` → input features only
- `sample_submission.csv` → required Kaggle submission format

###  Key Features
- `comment` → raw user text
- `created_date` → timestamp
- `post_id` → thread identifier
- `emoticon_1`, `emoticon_2`, `emoticon_3` → symbolic indicators
- `upvote`, `downvote` → engagement signals
- `if_1`, `if_2` → hidden internal features
- `race`, `religion`, `gender`, `disability` → sensitive topic indicators
- `label` → target variable (4 classes)

---

## ⚙️ Approach

### 1) Feature Engineering
- Applied **TF-IDF vectorization** using both **word and character n-grams**
- Created engagement-based features such as **vote ratio**
- Extracted time-based features from timestamps
- Encoded high-cardinality categorical features
- Combined text + structured features into a unified ML pipeline

### 2) Models Used
- **Logistic Regression** → strong baseline
- **Linear SVC** → high precision on sparse text data
- **LightGBM** → strong recall and non-linear feature handling

### 3) Final Model
Final predictions were generated using a **soft-voting weighted ensemble**:

- **70% LightGBM**
- **30% Linear SVC**

This improved robustness and overall leaderboard performance.

---

##  Results
- Achieved **0.83282 leaderboard score**
- Ranked **191 / 2744**
- Finished in the **Top 7%**
- Improved minority-class prediction through ensemble learning
- Achieved a strong balance between **precision, recall, and weighted F1-score**

---

##  Key Learnings
- **TF-IDF tuning** had the biggest impact on performance
- Linear models performed exceptionally well on sparse text features
- Boosting models improved recall on difficult minority classes
- Ensemble methods helped balance precision-recall trade-offs
- Feature engineering on metadata significantly boosted model stability

---

##  Tech Stack
- Python
- Scikit-learn
- LightGBM
- Pandas
- NumPy
- Matplotlib

---

##  Competition
**Kaggle — Comment Category Prediction Challenge**

---
