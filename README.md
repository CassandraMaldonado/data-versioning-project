# Data Versioning and Differential Privacy Project

## Project Description

This project was developed as part of my coursework to better understand how data versioning supports reproducible machine learning workflows. I compared DVC and Delta Lake using a dataset of CrossFit athletes and built a simple Random Forest regression model to predict the total lift weight.

To make the project more realistic, I also added a DP step using the Laplace Mechanism to see how privacy noise affects model performance. The goal was to gain practical experience with version control, data reproducibility and privacy tradeoffs.

## Methodology

- Version 1 (v1): Original dataset.

- Version 2 (v2): Cleaned version with outliers removed.

Both versions were tracked using DVC and Delta Lake, then used to train and compare Random Forest models. Differential privacy was simulated with the Laplace Mechanism (e = 2.5) to measure its effect on accuracy.

## Results
DVC vs Delta Lake
Criterion	DVC	Delta Lake
Setup	Requires Git and CLI configuration	Python API; easier setup
Versioning	File-based	Table-based (time travel)
Switching	Manual checkout	Instant query by version
Best Use	ML artifact tracking	Data lake versioning

Result: Delta Lake was faster and simpler for versioning tabular data, while DVC worked better for Git-integrated workflows.
