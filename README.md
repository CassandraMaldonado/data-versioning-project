# Data Versioning and Differential Privacy Project

## Project Description

This project demonstrates a MLOps workflow integrating open-source versioning tools and privacy-preserving techniques. Using a dataset of CrossFit athletes, I compared DVC and Delta Lake for reproducible dataset management, trained Random Forest regression models to predict total lift weight and introduced DP using the Laplace mechanism to evaluate the accuracy–privacy trade-off.

The workflow aligns with the full assignment specification from data cleaning and feature engineering to EDA, model evaluation, linting and DP implementation, all under a traceable version-controlled pipeline.


## Tools and Libraries  

| Tool / Library | Purpose |
|-----------------|----------|
| **DVC** | File-based version control integrated with Git. |
| **Delta Lake** | Table-based versioning and time-travel queries. |
| **Scikit-Learn** | Model training. |
| **Matplotlib / Seaborn** | EDA. |
| **Flake8** | Linter for code quality. |
| **Shutil** | Dataset switching automation. |


## Methodology

- Version 1 (v1): Original dataset.

- Version 2 (v2): Cleaned version with outliers removed.

Both versions were tracked using DVC and Delta lake, then used to train and compare Random Forest models. DP was simulated with the Laplace Mechanism (e = 2.5) to measure its effect on accuracy.

## Results

Delta lake was faster and simpler for versioning tabular data, while DVC worked better for Git-integrated workflows.

| Criterion | DVC | Delta Lake |
|------------|------|-------------|
| Setup | Requires Git and CLI configuration | Python API; easier setup |
| Versioning | File-based | Table-based (time travel) |
| Switching | Manual checkout | Instant query by version |
| Best Use | ML artifact tracking | Data lake versioning |

DP vs Non-DP

| Setting | RMSE | MAE | R² |
|----------|------|-----|----|
| Non-DP | 10.11 | 3.81 | 0.9987 |
| DP (Laplace, ε = 2.5) | 97.56 | 69.26 | 0.8756 |

## Key Findings

- Delta Lake offered smoother version management for data tables.

- Cleaning data improved model accuracy by around %.

- Applying Differential Privacy reduced accuracy slightly (%) but improved data protection.

- The Laplace Mechanism is an easy way to apply DP noise without complex model changes.

