# Data Versioning and Differential Privacy Project

## Project Description

This project was developed to better understand how data versioning supports reproducible machine learning workflows. I compared DVC and Delta Lake using a dataset of CrossFit athletes and built a Random Forest regression model to predict the total lift weight. I added a DP step using the Laplace mechanism (e = 2.5) to see how privacy noise affects model performance. The goal was to gain experience with version control, data reproducibility and privacy tradeoffs.

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

