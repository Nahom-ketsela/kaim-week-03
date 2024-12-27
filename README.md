# AlphaCare Insurance Solutions (ACIS) - Marketing Analytics Project

Welcome to the ACIS Marketing Analytics Project repository. This project focuses on analyzing historical car insurance claim data to optimize marketing strategies and identify low-risk customer segments. The project employs a mix of Exploratory Data Analysis (EDA), Statistical Modeling, and Machine Learning techniques.

## Introduction

The goal of this project is to leverage historical car insurance data (from February 2014 to August 2015) to:

- Improve marketing strategies.
- Identify low-risk customer segments for reduced premiums.
- Develop predictive models to optimize premiums and better understand customer behavior.

The project sharpens skills in Data Engineering (DE), Predictive Analytics (PA), and Machine Learning Engineering (MLE) while simulating real-world analytics workflows and challenges.


---

## Tasks Overview

### Git and GitHub Setup

**Objective:** Initialize version control and establish CI/CD pipelines.

**Key Steps:**
- Create a GitHub repository and initialize Git.
- Set up branches for each task (e.g., task-1, task-2).
- Implement GitHub Actions for CI/CD.
- Commit work regularly with meaningful messages.

### Data Version Control (DVC)

**Objective:** Implement DVC for data tracking and versioning.

**Key Steps:**
- Install and configure DVC.
- Track datasets and manage versions.
- Commit `.dvc` files to Git.
- Use local storage for DVC remote.

###  Statistical Testing

**Objective:** Validate hypotheses using statistical methods.

**Key Steps:**
- Segment data into groups (e.g., based on zip codes, gender).
- Perform chi-squared, t-tests, or z-tests to evaluate differences.
- Analyze p-values and document findings.

### Statistical Modeling

**Objective:** Build and evaluate machine learning models.

**Key Steps:**
- Prepare data: Handle missing values, engineer features, and encode categorical variables.
- Train models: Implement Linear Regression, Random Forests, and XGBoost.
- Evaluate models: Use metrics like RMSE, R-squared, and feature importance.
- Interpret results: Apply SHAP or LIME for model explainability.

---

## Requirements

To set up and run the project, install the following dependencies:

- Python 3.8+
- Required Python libraries (listed in `requirements.txt`)
- Git and GitHub CLI
- DVC
- Jupyter Notebook

Install dependencies using:

```bash
pip install -r requirements.txt

