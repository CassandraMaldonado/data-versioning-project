# Data Versioning and Differential Privacy Project

## Project Description

This project was developed to better understand how data versioning supports reproducible machine learning workflows. I compared DVC and Delta Lake using a dataset of CrossFit athletes and built a Random Forest regression model to predict the total lift weight.

To make the project more realistic, I also added a Differential Privacy (DP) step using the Laplace Mechanism (ε = 2.5) to see how privacy noise affects model performance. The goal was to gain practical experience with version control, data reproducibility, and privacy tradeoffs—all key concepts in responsible AI and data science.

## Methodology

- Version 1 (v1): Original dataset.

- Version 2 (v2): Cleaned version with outliers removed.

Both versions were tracked using DVC and Delta Lake, then used to train and compare Random Forest models. Differential privacy was simulated with the Laplace Mechanism (e = 2.5) to measure its effect on accuracy.

## Results

**Result:** Delta Lake was faster and simpler for versioning tabular data, while DVC worked better for Git-integrated workflows.
