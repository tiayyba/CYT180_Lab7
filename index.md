# CYT180 — Lab 7: Data Preprocessing for Machine Learning 
**Weight:** 3% <br>
**Work Type:** Individual <br>
**Submission Format:** screenshots.

----

## Introduction
In this lab, you will perform data preprocessing and exploratory data analysis (EDA) on a structured dataset. Data preprocessing and exploratory data analysis (EDA) are the foundation of every machine‑learning pipeline, especially in cybersecurity analytics where data can be noisy, inconsistent, incomplete, or heavily imbalanced.
Before building any model—whether for intrusion detection, anomaly detection, fraud scoring, or malware classification, you must thoroughly understand the raw data. Preprocessing ensures the dataset is clean, reliable, and mathematically prepared for ML algorithms. EDA helps reveal hidden patterns, relationships, suspicious behaviors, and structural issues long before modeling begins.
In cybersecurity, poorly preprocessed data can lead to:

- False positives in anomaly detection
- Missed threats due to skewed or unscaled features
- Misleading correlations
- Overfitting from extreme outliers
- Broken models due to missing values
- Incorrect assumptions about normal vs malicious behavior

This lab guides you through the full preprocessing pipeline from loading the data up to scaling, ensuring that you have a clean, trustworthy dataset ready for modeling in next lab.

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
Before swe get started, it is important to understand the difference between Data Preprocessing and Exploratory Data Analysis (EDA). Although related, they serve different roles in the machine-learning workflow.

### What Is Data Preprocessing?

Data preprocessing involves preparing raw data so it is clean, consistent, and ready for modeling.

**Key Characteristics**
- Improves data quality  
- Corrects structural issues  
- Ensures machine-learning algorithms can interpret the data  
- Handles inconsistencies and noise  

**Typical Preprocessing Tasks**
- Handling missing values  
- Treating or removing outliers  
- Fixing incorrect data types  
- Encoding categorical variables  
- Normalizing or standardizing values  
- Removing duplicates  
- Selecting or filtering features

### What Is Exploratory Data Analysis (EDA)?

Exploratory Data Analysis (EDA) focuses on understanding the dataset and uncovering patterns.

**Key Characteristics**
- Examines trends and distributions  
- Identifies relationships between variables  
- Detects anomalies or unexpected behavior  
- Helps guide preprocessing and modeling decisions  

**Typical EDA Tasks**
- Summary statistics (mean, median, quartiles)  
- Distribution visualizations  
- Correlation analysis  
- Checking class balance  
- Observing dataset structure and behavior  

#### How They Work Together

A typical ML workflow alternates between the two:

1. Perform EDA → discover outliers
2. Preprocess → remove or adjust outliers
3. EDA again → verify the effect
4. Preprocess → scale features
5. EDA again → check distribution changes

It’s an iterative loop, not a single pass.

----

## Section 1 — Import Libraries & Load Dataset
Before cleaning, analyzing, or modeling, confirm the dataset loads correctly, columns appear as expected, and no immediate formatting issues exist. This prevents downstream errors and saves time. Lets set up your Python environment and load the dataset to begin preprocessing and EDA.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv') #  loads the dataset
df.head() # previews the first 5 rows to verify structure and column names.
```

**Explanation**

- **pandas (pd):** DataFrame operations (load CSVs, filter, group, summarize).
- **numpy (np):** Numerical utilities (percentiles, array math, element-wise ops).
- **scikit-learn scalers:**
  - MinMaxScaler → normalize features to [0, 1].
  - StandardScaler → standardize to mean = 0, std = 1.
- **seaborn (sns) & matplotlib (plt):** Plotting and statistical visualization.

----

## Section 2: Inspect Structure and Check Missing Values

Inspect the dataset’s structure and identify missing values before performing any transformations.Missing or inconsistent data can:

- Distort statistical summaries
- Break machine-learning models
- Lead to incorrect conclusions

Identifying missing values early helps determine whether to remove rows, impute values, or engineer alternative solutions.

```python
df.info() # number of rows and columns, Data Typles, How many non-null entries each column contains
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
**Explanation**

- **df.describe()** provides a quick summary of each numeric column, including the mean, median (50%), quartiles, and spread.
- Boxplots highlight the distribution shape, central value, and any values beyond the typical 1.5 × IQR boundary.
- This step helps identify skew, inconsistent scales, and potential outliers before deciding whether to remove or transform them.

**Reflection Questions**
- Why might boxplots reveal outliers more effectively than numerical summaries alone?
- Should all outliers be removed automatically, or should the decision depend on domain context?
- How might skewed distributions influence future preprocessing steps?

----

## Section 4: Remove Outliers Using the IQR Method

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

**Explanation**

- The first quartile `(Q1)` and third quartile `(Q3)` define the middle 50 percent of the data.
- The Interquartile Range `(IQR)` is computed as `Q3 minus Q1`.
- Any value below `Q1 − 1.5 × IQR` or above `Q3 + 1.5 × IQR` is considered a potential outlier.
- This method uses a consistent, reproducible rule and is widely used for skewed or non‑normal data.

Clinical datasets (such as this diabetes dataset) may contain zeros or unusually high values that indicate missing or irregular entries.
In cybersecurity datasets, some outliers may represent meaningful anomalies rather than noise. These should be evaluated carefully before removal.

**Reflection Questions**

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
**Explanation**

- **df.corr()** computes pairwise correlation coefficients between numeric features.
- Correlation values range from:
  - +1.0 (strong positive relationship)
  - 0 (no linear relationship)
  - 1.0 (strong negative relationship)
- The heatmap color‑codes these relationships, making it easier to spot strong associations.

### Why This Step Matters (Key Rationale)
1. **Identifying Redundant Features:** Highly correlated features often provide overlapping information. Including both in a model can add noise or reduce efficiency without improving accuracy. Feature selection often starts with correlation inspection.
2. **Understanding Feature Influence on the Target:** A feature strongly correlated with the target variable is a good candidate for inclusion in a predictive model. Conversely, features with almost no correlation may be less useful or require transformation.
3. **Detecting Multicollinearity Before Modeling:** In models such as logistic regression, linear regression, or any model relying on coefficients, multicollinearity (when predictors are strongly correlated with each other) weakens model reliability. Correlation analysis helps you detect and address this early.
4. **Supporting Feature Engineering:** Correlations can reveal hidden relationships that guide decisions such as:
   - combining features
  - removing irrelevant fields
  - applying transformations
5. **Providing Insight Into Data Structure:** Before doing any modeling, analysts should understand how their variables interact. Correlation analysis provides a quick, evidence‑based overview of the dataset’s structure.
