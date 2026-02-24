# CYT180 — Lab 7: Data Preprocessing for Machine Learning 
**Weight:** 3% <br>
**Work Type:** Individual <br>
**Submission Format:** video

----

## Introduction
In this lab, you will perform data preprocessing and exploratory data analysis (EDA) on a structured dataset. Data preprocessing and exploratory data analysis (EDA) are the foundation of every machine‑learning pipeline, especially in cybersecurity analytics where data can be noisy, inconsistent, incomplete, or heavily imbalanced.Before building any model—whether for intrusion detection, anomaly detection, fraud scoring, or malware classification, you must thoroughly understand the raw data.<br>
Preprocessing ensures the dataset is clean, reliable, and mathematically prepared for ML algorithms. EDA helps reveal hidden patterns, relationships, suspicious behaviors, and structural issues long before modeling begins.
In cybersecurity, poorly preprocessed data can lead to:

- False positives in anomaly detection
- Missed threats due to skewed or unscaled features
- Misleading correlations
- Overfitting from extreme outliers
- Broken models due to missing values
- Incorrect assumptions about normal vs malicious behavior

This lab guides you through the full preprocessing pipeline from loading data to scaling (data normalization), ensuring that you have a clean, trustworthy dataset ready for modeling in next lab.

----

## Learning Objectives
By the end of this lab you will be able to:
- Load and inspect a dataset
- Identify and understand missing data
- Generate statistical summaries
- Detect and remove outliers
- Analyze correlations
- Examine class distributions
- Separate features and target variables
- Apply feature scaling

----

## Data Preprocessing vs. Exploratory Data Analysis (EDA)
Before we get started, it is important to understand the difference between Data Preprocessing and Exploratory Data Analysis (EDA). Although related, they serve different roles in the machine-learning workflow.

### What Is Data Preprocessing?

Data preprocessing involves preparing raw data so it is clean, consistent, and ready for modeling. It improves data quality, corrects structural issues, handle inconsistencies and noise and ensures that machine-learning algorithms can interpret the data.

#### Typical Preprocessing Tasks
- Handling missing values  
- Treating or removing outliers  
- Fixing incorrect data types  
- Encoding categorical variables  
- Normalizing or standardizing values  
- Removing duplicates  
- Selecting or filtering features

### What Is Exploratory Data Analysis (EDA)?

Exploratory Data Analysis (EDA) focuses on understanding the dataset and uncovering patterns. This process examines trends and distributions, identifies relationships between variables, detects anomalies or unexpected behavior, and helps guide preprocessing and modeling decisions.

#### Typical EDA Tasks
- Summary statistics (mean, median, quartiles)  
- Distribution visualizations  
- Correlation analysis  
- Checking class balance  
- Observing dataset structure and behavior  

### How They Work Together

A typical ML workflow alternates between the two:

1. Perform EDA → discover outliers
2. Preprocess → remove or adjust outliers
3. EDA again → verify the effect
4. Preprocess → scale features

It’s an iterative loop, not a single pass.

----

## Section 1 — Import Libraries & Load Dataset
Before cleaning, analyzing, or modeling, confirm the dataset loads correctly, columns appear as expected, and no immediate formatting issues exist. This prevents downstream errors and saves time. Let's set up your Python environment and load the dataset to begin preprocessing and EDA.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv') #  loads the dataset
df.head() # previews the first 5 rows to verify structure and column names.
```

### Explanation

- **pandas (pd):** DataFrame operations (load CSVs, filter, group, summarize).
- **numpy (np):** Numerical utilities (percentiles, array math, element-wise ops).
- **scikit-learn scalers:**
  - MinMaxScaler → normalize features to [0, 1].
  - StandardScaler → standardize to mean = 0, std = 1.
- **seaborn (sns) & matplotlib (plt):** Plotting and statistical visualization.

----

## Section 2: Inspect Structure and Check Missing Values

Inspect the dataset’s structure and identify missing values before performing any transformations. Missing or inconsistent data can:

- Distort statistical summaries
- Break machine-learning models
- Lead to incorrect conclusions

Identifying missing values early helps determine whether to remove rows, impute values, or engineer alternative solutions.

```python
df.info() # number of rows and columns, data types, how many non-null entries each column contains
df.isnull().sum()  # count of missing values per column
```
----

## Section 3: Statistical Summary and Outlier Visualization
Before performing any cleaning or transformations, it is important to understand how each numeric feature (column) is distributed. This section provides two key tools for this:
- **descriptive statistics:** which summarize the central tendency and spread of each feature
- **boxplots:** which help visually detect potential outliers

Descriptive statistics confirm whether feature values fall within expected ranges. Visualizing outliers helps avoid distortions during scaling or model training.


```python
summary = df.describe()  # count, mean, std, min, 25%, 50%, 75%, max
print(summary)

import matplotlib.pyplot as plt
num_cols = df.select_dtypes(include=["number"]).columns
fig, axs = plt.subplots(len(num_cols), 1, figsize=(7, 1.8 * len(num_cols)), constrained_layout=True)
if len(num_cols) == 1:
    axs = [axs]  # ensure iterable

for ax, col in zip(axs, num_cols):
    ax.boxplot(df[col], vert=False)
    ax.set_ylabel(col)
    ax.set_xlabel("Value")
    ax.set_title(f"Boxplot: {col}")

plt.show()
```
### Explanation

- **df.describe()** provides a quick summary of each numeric column, including the mean, median (50%), quartiles, and spread.
- Boxplots highlight the distribution shape, central value, and any values beyond the typical 1.5 × IQR boundary.
- This step helps identify skew, inconsistent scales, and potential outliers before deciding whether to remove or transform them.

### Reflection Questions
- Why might boxplots reveal outliers more effectively than numerical summaries alone?
- Should all outliers be removed automatically, or should the decision depend on domain context?
- How might skewed distributions influence future preprocessing steps?

----

## Section 4: Remove Outliers

Extreme values can distort statistical summaries, scaling procedures, and distance‑based algorithms. After identifying potential outliers in Step 3, the next step is to determine whether these values should be removed.  Removing or adjusting outliers helps maintain stable model training.
In some domains, outliers may represent true rare behavior; in others, they may indicate data entry errors or noise. The decision should always be justified.

One common and transparent approach is the Interquartile Range (IQR) method. This method uses statistical boundaries to identify values that fall far outside the typical range of the data. Outlier removal can improve model stability, especially when extreme values distort the scale or distribution of a feature.

```python
import numpy as np

# Example: Removing outliers from the "Insulin" column using the IQR method
q1, q3 = np.percentile(df["Insulin"].dropna(), [25, 75])
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

clean_df = df[(df["Insulin"] >= lower_bound) & (df["Insulin"] <= upper_bound)]

print("Original rows:", len(df))
print("Rows after removing Insulin outliers:", len(clean_df))
```

### Explanation

- The first quartile `(Q1)` and third quartile `(Q3)` define the middle 50 percent of the data.
- The Interquartile Range `(IQR)` is computed as `Q3 minus Q1`.
- Any value below `Q1 − 1.5 × IQR` or above `Q3 + 1.5 × IQR` is considered a potential outlier.
- This method uses a consistent, reproducible rule and is widely used for skewed or non‑normal data.

Clinical datasets (such as this diabetes dataset) may contain zeros or unusually high values that indicate missing or irregular entries.
In cybersecurity datasets, some outliers may represent meaningful anomalies rather than noise. These should be evaluated carefully before removal.

### Reflection Questions

- Why is 1.5 × IQR used as a standard boundary rather than 1 × IQR or 2 × IQR?
- Should all outliers be removed automatically, or should outlier handling depend on the meaning of the data?
- How might removing outliers affect the distribution of a feature?

----

## Section 5: Correlation Analysis
Correlation analysis helps determine how strongly different features relate to one another. Before building any machine‑learning model, it is important to understand which variables move together, which are unrelated, and which may be redundant. This step provides a structured way to evaluate feature relationships based on numerical evidence. For many models, especially linear models and distance‑based algorithms, correlation patterns directly influence performance and interpretability.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix for numeric columns
corr = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
```
### Explanation

- **df.corr()** computes pairwise correlation coefficients between numeric features.
- Correlation values range from:
  - +1.0 (strong positive relationship)
  - 0 (no linear relationship)
  - -1.0 (strong negative relationship)
- The heatmap color‑codes these relationships, making it easier to spot strong associations.

### Why This Step Matters
- Identifying Redundant Features: Highly correlated features often provide overlapping information. Including both in a model can add noise or reduce efficiency without improving accuracy. Feature selection often starts with correlation inspection.
- Understanding Feature Influence on the Target: A feature strongly correlated with the target variable is a good candidate for inclusion in a predictive model. Conversely, features with almost no correlation may be less useful or require transformation.
- Detecting Multicollinearity Before Modeling: In models such as logistic regression, linear regression, or any model relying on coefficients, multicollinearity (when predictors are strongly correlated with each other) weakens model reliability. Correlation analysis helps you detect and address this early.
- **Supporting Feature Engineering:** Correlations can reveal hidden relationships that guide decisions such as:
   - combining features
   - removing irrelevant fields
   - applying transformations
- **Providing Insight Into Data Structure:** Before doing any modeling, analysts should understand how their variables interact. Correlation analysis provides a quick, evidence‑based overview of the dataset’s structure.

### Example: Identifying Redundant Features

Suppose the correlation matrix shows the following relationships:

- Glucose and Insulin have a correlation of **0.82**
- BMI and SkinThickness have a correlation of **0.78**
- Age and Pregnancies have a correlation of **0.20**

From this, we can reason:

- **Glucose and Insulin (~0.82 correlation)**  
   These two features move strongly together. Including both may add little additional information to the model.  
   A common approach:
   - Keep the feature with fewer missing values.
   - Or keep the one that shows a stronger correlation with the target (Outcome).

- **BMI and SkinThickness (~0.78 correlation)**  
   These are moderately high and could introduce redundancy.  
   You might:
   - Retain both if a nonlinear model (e.g., Random Forest) is used.
   - Drop one if using a model sensitive to multicollinearity (e.g., Logistic Regression).

- **Age and Pregnancies (~0.20 correlation)**  
   Low correlation means no redundancy. Both features capture different information and should be kept.

### Reflection Questions

- Why might two variables be strongly correlated even if one does not directly cause the other?
- What issues might arise if two features have a correlation greater than 0.9?
- How can correlation analysis inform feature selection?

----
## Section 6: Visualize Target Variable Distribution
Before training any machine-learning model, it is important to understand how balanced or imbalanced the target variable is. Class distribution affects model performance, fairness, and evaluation metrics. Many algorithms assume roughly balance classes, and if one class is significantly underrepresented, the model may learn to favor the majority class. This section visualizes the frequency of each class in the dataset to inform decisions about resampling or using class‑balanced algorithms later.

```python
import matplotlib.pyplot as plt

# Count values in the target column
counts = df["Outcome"].value_counts()

# Pie chart representation
plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=counts.index,
    autopct="%.0f%%",
    shadow=True
)
plt.title("Outcome Distribution")
plt.show()
```
### Explanation

- `value_counts()` counts how many samples belong to each class (0 vs. 1 in the diabetes dataset).
- A pie chart visually shows the proportion of each category.
- This visualization makes it easy to determine whether one class dominates the dataset.
  
### Why This Step Matters

- **Understanding Class Imbalance:** If one class greatly outweighs the other, the model may default to predicting the majority class. If 80% of samples are class 0 and 20% are class 1, a model that always predicts 0 will appear to be “80% accurate” while completely failing to detect class 1.
- **Planning for Resampling Techniques**
  - Oversampling (e.g., SMOTE)
  - Undersampling
  - Class-weighted algorithms
- **Identifying Evaluation Challenges:** With imbalance, metrics like precision, recall, F1-score, and ROC-AUC become more meaningful than accuracy.
- **Ensuring Fairness and Reliability:** In cybersecurity, minority classes often represent the most critical events (e.g., intrusion attempts). Even if rare, missing them would have serious consequences.

A roughly even distribution suggests the dataset is balanced and simpler to model. A skewed distribution (e.g., 90/10 or 70/30) indicates the need for special handling. It is important to always check both percentages and absolute counts.

In this lab, we only visualize class imbalance. We do not apply balancing techniques yet because oversampling or undersampling must be performed after train–test splitting to avoid data leakage. Class balancing will be handled in next Lab during model training.

### Reflection Questions
- Based on the pie chart, what percentage of the samples belong to class 0 (No Diabetes) and class 1 (Diabetes)?  
- Why is accuracy not a reliable metric when the dataset is moderately imbalanced?
- Why are we only visualizing class imbalance in this lab and not applying balancing techniques yet?
- In cybersecurity datasets, why are low-frequency (minority) events—such as intrusion attempts or failed login spikes—often the most important to detect?

----

## Section 7: Separate Features and Target
Before training a machine-learning model, it is necessary to split the dataset into two components:  
- **X**: all predictor variables (the inputs)  
- **y**: the target variable (the output we want to predict)  

This separation ensures that models learn patterns only from the predictors and that the target variable is not accidentally included in the training data. Including the target variable inside X would make the model “learn” using information it should not have access to, leading to unrealistically high accuracy. Including the target variable inside X would make the model “learn” using information it should not have access to, leading to unrealistically high accuracy. Almost all machine‑learning workflows use this structure.


```python
# Separate predictors and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
```

### Explanation

- X contains all input features except the target.
- y contains only the target column (Outcome), where:
  - 0 = No Diabetes
  - 1 = Diabetes
- This separation is required for model training, scaling, and evaluation steps.
- Using .drop() ensures no accidental inclusion of the target in the feature set.

### Reflection Questions
- What problems might occur if the target column is accidentally included in X?
- In a cybersecurity scenario, what could X and y represent?

----

## Section 8: Feature Scaling (Normalization and Standardization)
Machine-learning algorithms often assume that all features operate on comparable numerical scales. When features vary widely in range, models that rely on distance, gradient calculations, or optimization can behave unpredictably. Feature scaling transforms all predictors into a consistent range or distribution, improving training stability and model performance. In this step, you will apply two common scaling techniques: Min‑Max Normalization and Standardization.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1) Min-Max Normalization (scales features to range 0–1)
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)

# 2) Standardization (mean = 0, standard deviation = 1)
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

print("Shape of normalized X:", X_normalized.shape)
print("Shape of standardized X:", X_standardized.shape)
```

### Explanation

- Min‑Max Normalization
  - Rescales values to a fixed range (typically 0–1).
  - Useful when the feature distribution is not assumed to be normal.
  - Sensitive to outliers, which is why outlier removal (Step 4) is done beforehand.
- Standardization (Z‑Score Scaling)
  - Centers data around mean 0 with standard deviation 1.
  - Preferred for algorithms that assume normally distributed data or use gradient‑based optimization.
 

Without scaling, a feature like “Glucose” (0–200) can overpower a feature like “Pregnancies” (0–15), skewing model behavior. Imagine two features:
- Glucose values range from 50 to 200
- SkinThickness values range from 5 to 50
If unscaled, a model may treat changes in Glucose as far more significant than changes in SkinThickness simply because the raw numbers are larger. After scaling, both features contribute proportionally to the model.
Many algorithms expect scaled input for optimal performance, including:
  - Logistic Regression
  - k‑Nearest Neighbors
  - Neural Networks
  - PCA
  

### Reflection Questions
- Why must scaling be applied after separating X and y?

----

## Conclusion

In this lab, you completed the full preprocessing and exploratory data analysis workflow required to prepare a structured dataset for machine learning. You examined the dataset’s structure, identified missing values, explored statistical summaries, visualized outliers, applied the IQR method, analyzed feature correlations, inspected class balance, separated predictors from the target variable, and scaled the features.

These steps ensure the data is clean, consistent, and ready for modeling. By the end of this lab, you should understand not only how to perform these tasks, but also why each step matters in building reliable and interpretable machine‑learning models. This foundation is essential before moving into model training, evaluation, and prediction.

Your dataset is now fully preprocessed and ready for next **Lab 8**, where you will perform:

- Train–test splitting  
- Model selection  
- Model training  
- Model evaluation and reporting  
- Interpretation of performance metrics  

Having clean and well‑prepared data will make the modeling process in Lab 8 far more effective and meaningful.

## Submission Instructions
- Record a 3-minutes video where you show your notebook and explain your preprocessing and EDA verbally.
- The video must include these four checkpoints in order:
- **Checkpoint A — Dataset Loading and Initial Inspection: Section 1 – Section 2 (≤ 30 seconds)**
  - Show how you imported the required libraries, loaded the dataset, and used commands like `df.head()`, `df.info()`, and `df.isnull().sum()` to confirm the structure and identify any missing values.
- **Checkpoint B — Descriptive Statistics and Outlier Visualization: Section 3 – Section 4  (≤ 60 seconds)**
  - Explain how you generated summary statistics with `df.describe()` and used boxplots to visually detect potential outliers. Briefly describe what patterns or anomalies you observed
- **Checkpoint C — Correlation Analysis and Class Distribution: Section 5 – Section 6 (≤ 45 seconds)**
  - Show your correlation heatmap and discuss any strongly related features. Then present your class distribution visualization and comment on the imbalance (approximately 65% vs 35%) and why balancing is not performed in this lab.
- **Checkpoint D — Feature/Target Separation and Scaling: Section 7 – Section 8  (≤ 45 seconds)**
  - Demonstrate how you separated the dataset into **X** (features) and **y** (target). Then show the scaling step (Min‑Max or Standardization) and briefly explain why scaling is important before model training.
      
----

## Video Requirements

- Max length: 3 minutes (over 3 minutes = -20% deduction per minute)
- Screen share showing your notebook
- Voice narration required along with camera on
- One continuous video capture (no editing)
- Submit as: Unlisted YouTube link
- Paste your video link: in the Blackboard Lab 7 submission
