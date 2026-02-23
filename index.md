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

| Data Preprocessing | Exploratory Data Analysis (EDA) |
|--------------------|-----------------------------------|
| Prepares and cleans data | Helps you understand the data |
| Fixes problems | Reveals patterns and issues |
| Ensures data is model-ready | Guides preprocessing choices |
| Structured steps | Open-ended investigation |


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
