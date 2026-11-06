# CYT180 — Lab 7: Preparing Security Data for Machine Learning

**Weight:** 3%
**Work Type:** Individual
**Submission Format:** Your Colab notebook printed to PDF (see Submission Guidelines)

---

## Introduction

In this lab, you will prepare a network security dataset for machine learning. Before anyone can train a model to detect intrusions, scans, or data theft, the raw data has to be checked, cleaned, and put into a form that an algorithm can use. Security data is often noisy, incomplete, duplicated, and heavily imbalanced, because attacks are rare compared to normal traffic.

Poorly prepared data leads to real problems in security analytics:

- False positives in anomaly detection
- Missed threats because a feature was skewed or not scaled
- Misleading correlations
- Models that break because of missing values
- Wrong assumptions about what normal and malicious behaviour look like

You will work through the full preparation workflow, from loading the data to scaling the features. At the end, you will have a clean dataset that is ready for model training in Lab 8.

You will mostly run provided code and observe the results, and you will complete one small task in most sections.

**Every student works with their own sample of the data.** Your sample, and the column you use for the outlier section, are chosen by your student ID, so your numbers will be different from your classmates' numbers. Submissions with another student's numbers will receive 0.

---

## Learning Objectives

By the end of this lab, you will be able to:

- Load and inspect a security dataset with Pandas
- Find missing values and duplicate records, and handle them
- Read a statistical summary and boxplots
- Detect outliers with the IQR method and decide whether to remove them
- Analyze correlations between features and with the target
- Examine the class distribution (normal vs attack)
- Separate features (X) from the target (y)
- Apply Min-Max normalization and standardization

---

## Data Preprocessing vs. Exploratory Data Analysis (EDA)

These two activities are related, but they have different jobs.

**Data preprocessing** prepares raw data so it is clean, consistent, and ready for modeling. Typical tasks are handling missing values, removing duplicates, treating outliers, fixing data types, encoding categorical values, and scaling features.

**Exploratory Data Analysis (EDA)** is about understanding the data and finding patterns. Typical tasks are summary statistics, distribution plots, correlation analysis, and checking class balance.

In practice, you move back and forth between the two:

1. EDA: you discover outliers
2. Preprocessing: you decide what to do with them
3. EDA again: you check the effect
4. Preprocessing: you scale the features

It is a loop, not a single pass.

---

## About the Data

The file `network_connections.csv` contains **network connection records** from a simulated company network. Each row is **one connection** to a company server, summarized the way a network monitoring tool (such as Zeek or a firewall) would record it. The data was generated for this course, so it is safe to share and every student can get a different sample of it.

About 10% of the connections are **attacks** of three kinds:

- **Port scans:** one source quickly tries hundreds of ports to find open services
- **Brute-force logins:** many password attempts against one account
- **Data exfiltration:** a long connection that sends a very large amount of data out of the network

| Column | Meaning |
|---|---|
| `conn_id` | an ID number for the connection |
| `duration_sec` | how long the connection lasted, in seconds |
| `bytes_sent` | bytes sent from the inside host to the outside |
| `bytes_received` | bytes received by the inside host |
| `packets` | total packets in the connection |
| `failed_logins` | failed login attempts during the connection |
| `conn_per_min` | how many connections the same source made in that minute |
| `unique_ports` | how many different ports the same source contacted in that minute |
| `hour_of_day` | hour the connection started (0 to 23) |
| `label` | the target: **0 = normal**, **1 = attack** |

The data has the same problems that real logs have: some values are missing, some records were sent twice by the log forwarder, and some values are extremely large.

---

## Prerequisites

- You will work in **Google Colab**. Pandas, NumPy, scikit-learn, seaborn, and matplotlib are already installed there.
- Download `network_connections.csv` from Blackboard.
- In Colab, click the **folder icon** on the left, then the **upload** button, and choose the file. It will be saved as `/content/network_connections.csv`.

**Note:** files uploaded to Colab are deleted when the runtime restarts. If you get a "file not found" error, upload the file again.

---

## Grading (10 marks)

| Item | Marks |
|---|---|
| Task 1 – Missing values and duplicates | 1 |
| Task 2 – Fill the missing values | 1 |
| Task 3 – Outliers in your column | 1 |
| Task 4 – Correlation with the label | 1 |
| Task 5 – Class distribution | 1 |
| Task 6 – Check the scaled values | 1 |
| Reflection questions R1–R4 (1 mark each) | 4 |
| **Total** | **10** |

The provided code cells are not graded on their own, but your notebook must show that you ran them. A task whose output is missing receives 0. A submission without the Section 0 cell, or with numbers that do not match your student ID, receives 0.

---

## Section 0 — Your Name, Student ID, and Your Sample

Run this cell **first**, at the top of your notebook. Type your own full name and student ID between the quotes before running it.

```python
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

STUDENT_NAME = "Your Full Name"    # <-- type your full name
STUDENT_ID   = "000000000"         # <-- type your 9-digit student ID

if STUDENT_ID == "000000000" or STUDENT_NAME == "Your Full Name":
    raise ValueError("Type your own name and student ID in this cell first.")

# The last digit of your student ID picks the column you will check for outliers in Section 5
OUTLIER_COLUMNS = ["duration_sec", "bytes_sent", "bytes_received", "conn_per_min", "unique_ports"]
MY_COLUMN = OUTLIER_COLUMNS[int(STUDENT_ID.strip()[-1]) % 5]

print("Name:      ", STUDENT_NAME)
print("Student ID:", STUDENT_ID)
print("My column: ", MY_COLUMN)
print("Timestamp: ", datetime.datetime.now().isoformat())
```

**Explanation**

- **pandas (pd):** works with tables of data (DataFrames): load CSV files, filter rows, and summarize columns.
- **numpy (np):** number utilities, such as percentiles.
- **seaborn (sns) and matplotlib (plt):** charts and plots.
- **MinMaxScaler and StandardScaler:** the two scikit-learn tools you will use in Section 9 to scale features.

Use the variables `STUDENT_ID` and `MY_COLUMN` in later cells exactly as shown. Do not change them.

---

## Section 1 — Load the Dataset

First, confirm that the file loads correctly and the columns look as expected. Each student takes a random sample of 5,000 rows, and your student ID decides which rows you get.

```python
df_all = pd.read_csv("/content/network_connections.csv")

# Your own sample of 5,000 rows. Your student ID decides which rows you get.
df = df_all.sample(n=5000, random_state=int(STUDENT_ID)).reset_index(drop=True)

print("Rows in the full file:", len(df_all))
print("Rows in my sample:   ", len(df))
df.head()
```

**Explanation**

- `pd.read_csv()` reads the CSV file into a DataFrame.
- `sample(n=5000, random_state=...)` picks 5,000 random rows. The `random_state` is a seed: the same seed always gives the same rows, so if you run the cell again you get the same sample.
- `head()` shows the first 5 rows so you can check the column names and values.

---

## Section 2 — Inspect the Structure, Missing Values, and Duplicates

Before changing anything, look at what you have. Missing values can distort statistics and break most machine-learning models. Duplicate records count the same event twice, which gives it too much weight. In security logs, duplicates often come from a log forwarder that sent the same event more than once.

```python
df.info()                 # rows, columns, data types, and non-null counts
print(df.isnull().sum())  # number of missing values in each column
```

**Explanation**

- `info()` lists each column with its data type and how many values are **not** missing. If a column has fewer non-null values than the number of rows, it has missing values.
- `isnull().sum()` counts the missing values in each column.
- `duration_sec` and `bytes_received` are stored as `float64`. Pandas uses decimals for any column with missing values, because an empty value (`NaN`) is a special decimal value.

### Task 1 — Missing Values and Duplicates (1 mark)

Write code that prints:

1. the total number of missing values in your sample
2. the number of duplicate rows in your sample

Hint: `df.isnull().sum().sum()` adds up the missing values of all columns, and `df.duplicated().sum()` counts rows that are an exact copy of an earlier row.

```python
# TODO: Task 1 - print the total missing values and the number of duplicate rows
# Your code here

print("Task 1 |", STUDENT_ID)
```

---

## Section 3 — Handle Missing Values and Duplicates

Now fix the two problems you found. Duplicate rows are removed. Missing values are **filled** with the **median** of the column instead of being deleted, so you do not lose the rest of the information in those rows.

**Why the median and not the mean?** Columns like `duration_sec` are heavily skewed: most connections are short, but a few last over an hour. Those few long connections pull the mean far up, so the mean is not a typical value. The median is the middle value, so a few extreme values do not move it much.

```python
print("Rows before removing duplicates:", len(df))
df_clean = df.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates: ", len(df_clean))

# Fill missing durations with the median duration
print("Mean duration:  ", round(df_clean["duration_sec"].mean(), 1))
print("Median duration:", df_clean["duration_sec"].median())
df_clean["duration_sec"] = df_clean["duration_sec"].fillna(df_clean["duration_sec"].median())
```

**Explanation**

- `drop_duplicates()` keeps the first copy of each row and removes the exact copies.
- `fillna(value)` replaces every missing value in the column with `value`.
- Compare the mean and the median that are printed. The difference shows how much a few long connections pull the mean.

### Task 2 — Fill the Missing Values (1 mark)

`bytes_received` still has missing values. Fill them with the median of `bytes_received`, the same way as above. Then print the number of missing values left in each column of `df_clean`. Every column should show 0.

```python
# TODO: Task 2 - fill missing bytes_received with its median, then print df_clean.isnull().sum()
# Your code here

print("Task 2 |", STUDENT_ID)
```

From now on, use `df_clean` in every section.

---

## Section 4 — Statistical Summary and Boxplots

Before deciding what to do with extreme values, look at how each numeric column is spread out. A statistical summary gives the numbers, and a boxplot shows them as a picture.

```python
features = df_clean.drop(columns=["conn_id", "label"])   # conn_id is only an ID, and label is the target

print(features.describe().round(1))

fig, axs = plt.subplots(len(features.columns), 1, figsize=(7, 1.8 * len(features.columns)), constrained_layout=True)
for ax, col in zip(axs, features.columns):
    ax.boxplot(features[col], vert=False)
    ax.set_ylabel(col, rotation=0, ha="right")
plt.show()
```

**Explanation**

- `describe()` shows, for each column: count, mean, standard deviation (std), minimum, the 25th percentile (Q1), the median (50%), the 75th percentile (Q3), and the maximum.
- In a boxplot, the **box** goes from Q1 to Q3, so it holds the middle 50% of the values. The line inside the box is the median. The **whiskers** reach out to the most extreme values that are still within 1.5 × IQR of the box. Every point drawn beyond the whiskers is a potential **outlier**.
- Look at `bytes_sent`: the median is a few thousand bytes, but the maximum is tens of millions. The box is squeezed against the left side because a few values are so far out.

---

## Section 5 — Outliers: Find Them, Then Decide

Extreme values can distort statistics, scaling, and distance-based algorithms. A common and transparent way to find them is the **Interquartile Range (IQR)** method:

- **IQR = Q3 − Q1**, the width of the middle 50% of the data
- Any value below **Q1 − 1.5 × IQR** or above **Q3 + 1.5 × IQR** is flagged as an outlier

The 1.5 multiplier is a convention: it flags values that are clearly far from the middle without flagging too many normal ones. It is the same rule the boxplot whiskers use.

In many datasets, outliers are data-entry errors or noise, and removing them is the right choice. **In security data, outliers are often the attacks themselves.** A port scan touches hundreds of ports, and data exfiltration sends millions of bytes. So before removing anything, check what you would be removing.

The code below uses **your** column, `MY_COLUMN`, from Section 0.

```python
q1, q3 = np.percentile(df_clean[MY_COLUMN], [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

is_outlier = (df_clean[MY_COLUMN] < lower_bound) | (df_clean[MY_COLUMN] > upper_bound)
outliers = df_clean[is_outlier]

print("Column:", MY_COLUMN)
print("Q1 =", q1, " Q3 =", q3, " IQR =", iqr)
print("Lower bound =", lower_bound, " Upper bound =", upper_bound)
print("Rows flagged as outliers:", len(outliers))
```

**Explanation**

- `np.percentile(..., [25, 75])` returns Q1 and Q3.
- `is_outlier` is True for every row whose value is outside the two bounds. The `|` symbol means "or".
- `outliers` holds only the flagged rows.

### Task 3 — Outliers in Your Column (1 mark)

Using the `outliers` DataFrame, print:

1. how many of the flagged rows are **attacks** (`label` = 1)
2. how many attacks there are in all of `df_clean`

Hint: because `label` is 0 or 1, `outliers["label"].sum()` counts the attacks.

```python
# TODO: Task 3 - print the number of attacks among the outliers, and the number of attacks in df_clean
# Your code here

print("Task 3 |", STUDENT_ID)
```

**R1 (1 mark):** Using your Task 3 numbers, what percentage of the attacks in `df_clean` would be deleted if you removed the outliers in your column? Based on this, should we remove these outliers before training an intrusion detection model? Explain in 2–3 sentences, using your own numbers.

**Our decision for this lab:** we **keep** the outlier rows. The goal of the model in Lab 8 is to find attacks, and many attacks are the extreme values. Instead of deleting them, we will let the scaling step and the choice of model deal with large values.

---

## Section 6 — Correlation Analysis

Correlation measures how strongly two numeric columns move together. Values range from **+1** (they rise together), through **0** (no linear relationship), to **−1** (one rises as the other falls).

Correlation helps in two ways before modeling:

- **Redundant features:** two features with a very high correlation carry almost the same information. Keeping both adds little, and models that use coefficients, such as logistic regression, become unstable when two inputs are almost copies of each other (this is called **multicollinearity**).
- **Useful features:** a feature that is strongly correlated with the target (`label`) is likely to help the model separate attacks from normal traffic.

```python
corr = df_clean.drop(columns=["conn_id"]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
```

**Explanation**

- `corr()` computes the correlation between every pair of numeric columns. We drop `conn_id` first because an ID number has no meaning as a quantity.
- The heatmap colours each value, so strong relationships (dark red or dark blue) stand out. The diagonal is always 1.00, because every column is perfectly correlated with itself.

### Task 4 — Correlation with the Label (1 mark)

Print the correlation of every column with `label`, sorted from highest to lowest.

Hint: `corr["label"]` selects one column of the correlation matrix, and `.sort_values(ascending=False)` sorts it.

```python
# TODO: Task 4 - print corr["label"] sorted from highest to lowest
# Your code here

print("Task 4 |", STUDENT_ID)
```

**R2 (1 mark):** From your heatmap, which pair of **features** (not including `label`) has the highest correlation, and what is its value? Explain why these two columns move together, and say which one you would keep for a logistic regression model and why. Then name the feature most correlated with `label` from your Task 4 output, and explain in one sentence why it makes sense for the attacks in this dataset.

---

## Section 7 — Class Distribution (Normal vs Attack)

Before training a model, check how many examples belong to each class. If one class is much larger, a model can learn to always predict that class and still look accurate.

```python
counts = df_clean["label"].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(counts, labels=["Normal (0)", "Attack (1)"], autopct="%.1f%%")
plt.title("Normal vs Attack Connections")
plt.show()
```

**Explanation**

- `value_counts()` counts how many rows have each value of `label`. The larger class comes first, which here is 0 (normal).
- The pie chart shows each class as a share of the whole.

We only look at the imbalance in this lab. We do **not** balance the classes yet: oversampling or undersampling must be done **after** the train–test split, otherwise copies of the same attack can end up in both the training and the test set (data leakage). You will handle this in Lab 8.

### Task 5 — Class Distribution (1 mark)

Print the **number** of rows in each class and the **percentage** of rows in each class.

Hint: `value_counts(normalize=True)` returns fractions instead of counts. Multiply by 100 for percentages.

```python
# TODO: Task 5 - print the count and the percentage of each class
# Your code here

print("Task 5 |", STUDENT_ID)
```

**R3 (1 mark):** Using your Task 5 numbers: if a model predicted "normal" for every connection, what accuracy would it get on your data? Explain why this model would be useless for a security team, even though its accuracy looks high. (2–3 sentences)

---

## Section 8 — Separate Features (X) and Target (y)

A model learns from the input columns (**X**) to predict the target (**y**). The target must not be inside X: if it were, the model would simply read the answer and look perfect in testing, then fail on real traffic where the answer is unknown.

We also remove `conn_id`. It is just a number given to each connection, so it carries no information about whether a connection is an attack. A model could still find accidental patterns in it.

```python
X = df_clean.drop(columns=["conn_id", "label"])
y = df_clean["label"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Feature columns:", list(X.columns))
```

**Explanation**

- `drop(columns=[...])` returns a copy of the DataFrame without those columns. X has 8 feature columns.
- `y` is a single column with 0 for normal and 1 for attack.
- The number of rows in X and y must be the same, because each row of X belongs to one value in y.

---

## Section 9 — Feature Scaling (Normalization and Standardization)

The features in X are on very different scales. `hour_of_day` goes from 0 to 23, but `bytes_sent` goes up to tens of millions. Models that use distances or coefficients (k-nearest neighbours, logistic regression, neural networks, PCA) would treat `bytes_sent` as far more important, only because its numbers are bigger. Scaling puts all features on a comparable scale.

```python
# 1) Min-Max Normalization: every feature is rescaled to the range 0 to 1
minmax_scaler = MinMaxScaler()
X_normalized = pd.DataFrame(minmax_scaler.fit_transform(X), columns=X.columns)

# 2) Standardization: every feature gets mean 0 and standard deviation 1
standard_scaler = StandardScaler()
X_standardized = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)

print(X_normalized.describe().round(3))
print(X_standardized.describe().round(3))
```

**Explanation**

- **Min-Max normalization** uses (value − min) / (max − min). The smallest value becomes 0 and the largest becomes 1. Because it depends on the minimum and maximum, a single extreme value changes the result for every row.
- **Standardization (z-score)** uses (value − mean) / standard deviation. Each feature ends up with mean 0 and standard deviation 1. It has no fixed range.
- `fit_transform()` does two steps: **fit** learns the values it needs from the data (min and max, or mean and std), and **transform** applies the formula.
- We wrap the result in `pd.DataFrame(..., columns=X.columns)` so the column names are kept.

**Important for Lab 8:** in this lab, we fit the scalers on all rows to see how scaling works. In a real model, you split the data first and fit the scaler on the **training data only**, then use it to transform the test data. Fitting on all rows lets information from the test set leak into training.

### Task 6 — Check the Scaled Values (1 mark)

For **your column** (`MY_COLUMN`), print:

1. the minimum, median, and maximum of `X_normalized[MY_COLUMN]`
2. the mean and standard deviation of `X_standardized[MY_COLUMN]`, rounded to 3 decimals

```python
# TODO: Task 6 - print min, median, max of X_normalized[MY_COLUMN], and mean and std of X_standardized[MY_COLUMN]
# Your code here

print("Task 6 |", STUDENT_ID)
```

**R4 (1 mark):** Look at the median of your column after Min-Max normalization. Is it close to 0, close to 0.5, or close to 1? Explain why, using what you saw about outliers in Sections 4 and 5. What does this tell you about using Min-Max normalization on security data with extreme values? (2–3 sentences)

---

## Conclusion

In this lab, you prepared a network security dataset for machine learning. You inspected its structure, found and handled missing values and duplicate records, read a statistical summary and boxplots, used the IQR method to find outliers and saw that many of them are attacks, analyzed correlations, checked the class imbalance, separated the features from the target, and scaled the features.

Your data is now ready for Lab 8, where you will:

- Split the data into training and test sets
- Choose and train a model
- Evaluate it with metrics that work for imbalanced data
- Interpret the results

---

## Submission Guidelines

1. **Run the whole notebook from top to bottom** (Runtime → Run all) so every output is visible. The Section 0 cell must be the first cell.
2. **Write the reflection answers (R1–R4) in a text cell** directly under the task they belong to. Answer in your own words and refer to **your own numbers**. Delete any template or AI-generated text.
3. **Print the notebook to PDF:** File → Print → Save as PDF. Check that the outputs and charts are readable and not cut off.
4. **Submit the PDF** on Blackboard, named `Lab7_FirstName_LastName.pdf`.

Deductions:

- A missing task output or missing reflection answer receives 0 for that item.
- Missing Section 0 cell, or numbers that do not match your student ID: 0 for the lab.
