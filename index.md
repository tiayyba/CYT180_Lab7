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
- **boxplots:** which help visually detect potential outliers.

Descriptive statistics confirm whether feature values fall within expected ranges. Visualizing outliers helps avoid distortions during scaling (Step 8) or model training


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
