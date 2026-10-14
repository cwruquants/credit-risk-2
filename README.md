# Credit Risk Modeling: Probability of Default

## Project Overview

This project introduces students to machine learning and quantitative finance through the development of a credit risk modeling pipeline. Using a credit risk dataset from Kaggle, participants will explore how different statistical and machine learning models can be used to estimate the **probability of default (PD)** for a borrower.

The project is designed for freshmen and other underclassmen with limited programming, machine learning, or financial experience. Participants will progress from exploratory data analysis and basic statistical modeling to building a structured, reproducible machine learning codebase.

The primary goal is **to learn the modeling process and software development workflow**, rather than to achieve the highest possible predictive accuracy.

## Learning Objectives

By the end of the project, participants should have experience with:

- **Python:** Basic syntax, data manipulation, and common data science libraries.
- **Git and GitHub:** Repositories, branching, commits, pushing, pulling, and pull request reviews.
- **Development environments:** Using VS Code, Jupyter Notebooks, or Google Colab.
- **Data analysis:** Data cleaning, exploratory data analysis, feature engineering, and correlation analysis.
- **Statistical modeling:** Logistic regression, model evaluation, and cross-validation.
- **Financial concepts:** Credit risk, probability of default, and the fundamentals of risk modeling.
- **Machine learning workflows:** Training, evaluating, saving, and running inference with models.
- **Software development:** Organizing code into reusable modules and maintaining a clean, version-controlled codebase.

## Dataset

We will use a credit risk dataset from Kaggle containing borrower characteristics and credit-related information. The dataset will serve as the basis for predicting whether a borrower defaults.

Participants will investigate the available features, identify the target variable, and determine which preprocessing steps are necessary before modeling.

*The specific dataset and its documentation will be selected before the project begins.*

## Project Roadmap

### Phase 1: Development Environment and Version Control

**Estimated duration:** 1–2 weeks

Establish the basic programming tools and collaboration workflow needed for the rest of the project.

Topics:
- Introduction to Python development in VS Code.
- Using Jupyter Notebooks or Google Colab for interactive development.
- Introduction to Git and GitHub.
- Basic Git commands and workflows:
  - Cloning repositories.
  - Creating branches and commits.
  - Pulling and pushing changes.
  - Opening and reviewing pull requests.
- Installing Python packages with `pip`.
- Introduction to virtual environments and dependency management.

Participants may use AI coding assistants to help write and understand code. The emphasis is on learning how to run, debug, and modify code rather than writing everything from scratch.

### Phase 2: Exploratory Data Analysis and Proof of Concept

**Estimated duration:** 1–2 weeks

Use Jupyter Notebooks to explore the dataset and build an initial probability of default model.

Topics:
- Loading and inspecting the dataset with pandas.
- Understanding feature types and the target variable.
- Identifying missing values, duplicates, outliers, and potential data quality issues.
- Performing exploratory data analysis and visualizing feature distributions.
- Examining correlations and relationships between features.
- Performing basic feature engineering and preprocessing.
- Splitting data into training and test sets.
- Training an initial logistic regression model.
- Evaluating predictions using appropriate classification and probability-based metrics.

The goal is to establish a working baseline and understand the reasoning behind each modeling step before introducing more complex algorithms.

### Phase 3: Building a Reusable Machine Learning Pipeline

**Estimated duration:** 2–3 weeks

Move beyond standalone notebooks and develop a structured codebase within the CWRU Quants GitHub organization.

A possible starting structure is:

```text
credit-risk/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data/
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   └── visualize.py
├── models/
├── tests/
├── requirements.txt
└── README.md
```

Topics:
- Separating data processing, training, inference, evaluation, and visualization.
- Refactoring notebook code into reusable Python modules.
- Implementing a repeatable training pipeline.
- Saving trained model artifacts and loading them for inference.
- Using cross-validation, including k-fold validation where appropriate.
- Comparing training, validation, and test performance.
- Visualizing model performance and predicted probabilities.
- Collaborating through branches and pull requests.
- Maintaining a reproducible and organized codebase.

The emphasis is on understanding how a maintainable machine learning project is structured and how multiple contributors can collaborate effectively.

## Advanced Extensions

Participants who complete the core project or want to explore additional topics can pursue the following extensions.

### 1. Environment and Command-Line Tooling

Develop a more robust local Python development environment using tools such as `uv` or `pyenv`.

Create command-line interfaces (CLIs) for common tasks, such as:

```bash
python -m src.train --model logistic
python -m src.infer --model logistic
python -m src.evaluate --model logistic
python -m src.visualize --model logistic
```

The exact interface can be adapted to the project's implementation.

### 2. Model Comparison Framework

Train and compare multiple model families, such as:

- Logistic regression.
- Decision trees and random forests.
- Gradient-boosted trees.
- Neural networks.

Develop a consistent framework for saving model artifacts, generating predictions, and comparing models using the same evaluation procedures.

Potential comparisons include:
- ROC and precision-recall curves.
- Calibration curves.
- Confusion matrices at selected decision thresholds.
- Predicted probability distributions.
- Discrimination and calibration metrics.

The objective is to understand the strengths, limitations, and trade-offs of different modeling approaches.

### 3. Prediction Uncertainty and Confidence Intervals

Explore methods for quantifying uncertainty in model predictions.

Possible approaches include:
- Quantile regression for estimating conditional outcome quantiles.
- Bootstrap methods for assessing model and prediction variability.
- Calibration and uncertainty analysis for predicted probabilities.

Visualize uncertainty alongside model predictions and investigate how it changes across borrower groups or risk levels.

*Note: Quantile regression estimates conditional quantiles of an outcome; it does not automatically produce confidence intervals for default probabilities. The method should be chosen based on the specific quantity we want to estimate.*

### 4. Model Artifact Management

Develop a system for storing and tracking trained models so that different artifacts can be compared consistently.

For example:

```text
models/
├── logistic_regression/
├── random_forest/
├── gradient_boosting/
└── neural_network/
```

Each artifact should retain relevant metadata, such as:
- Model type and hyperparameters.
- Training dataset and data split.
- Validation metrics.
- Feature preprocessing configuration.
- Training date and model version.

Use this information to generate reproducible comparisons across models and experiments.

## Evaluation and Success Criteria

The project will be considered successful when participants can:

1. Explain the basic credit risk modeling problem and the meaning of probability of default.
2. Inspect, clean, and prepare a dataset for modeling.
3. Train and evaluate a baseline logistic regression model.
4. Explain the purpose of training, validation, and test datasets.
5. Run a repeatable training and inference pipeline.
6. Use GitHub to collaborate on a shared codebase.
7. Compare different models using consistent evaluation metrics.

Advanced participants can extend the project with additional algorithms, uncertainty estimation, and command-line tooling.

## Important Modeling Considerations

Credit risk modeling requires particular attention to data leakage, class imbalance, probability calibration, and evaluation methodology.

- **Data leakage:** Features must not reveal information that would be unavailable when assessing a borrower at the time of prediction.
- **Class imbalance:** Default events may be relatively uncommon, so accuracy alone may be misleading.
- **Probability calibration:** A model's predicted probabilities should be evaluated to determine whether they correspond to observed default frequencies.
- **Validation:** Preprocessing and feature selection must be performed without leaking information from held-out data.
- **Evaluation:** Metrics should reflect the modeling objective and the costs associated with false positives and false negatives.

These considerations will be introduced as the project progresses.

## Getting Started

1. Install Python and a development environment such as VS Code.
2. Create a GitHub account and become familiar with basic Git commands.
3. Obtain the selected Kaggle dataset and review its documentation.
4. Complete the initial exploratory analysis notebook.
5. Build and evaluate a baseline logistic regression model.
6. Gradually transition the notebook workflow into the shared project codebase.

No prior experience in quantitative finance or machine learning is required. Curiosity, consistent participation, and a willingness to learn are the most important prerequisites.
