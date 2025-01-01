# Patient Selection for Diabetes Drug Testing
## Overview 
Creation of an Expected Hospitalization Time Regression and Uncertainty Estimation Model using EHR data

## Skills demonstrated
* Usage of Tensorflow Dataset API to scalably extract, transform, and load datasets and build datasets aggregated at the line, encounter, and patient data levels(longitudinal)
* Analyzed EHR datasets to check for common issues (data leakage, statistical properties, missing values, high cardinality) by performing exploratory data analysis.
* Created categorical features from Key Industry Code Sets (ICD, CPT, NDC) and reduce dimensionality for high cardinality features by using embeddings
* Created derived features(bucketing, cross-features, embeddings) utilizing Tensorflow feature columns on both continuous and categorical input features
* SWBAT use the Tensorflow Probability library to train a model that provides uncertainty range predictions that allow for risk adjustment/prioritization and triaging of predictions
* Analyzed and determined biases for a model for key demographic groups by evaluating performance metrics across groups by using the Aequitas framework
