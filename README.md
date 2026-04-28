# Coursework for the IBM - Machine Learning Coursera Track

**[Course Link](https://www.coursera.org/professional-certificates/ibm-machine-learning)**

- Datasets are not provided in this repository. They can be obtained by following the course.
- The [Capstone](./6-capstone-project/) project has a separate Poetry environment due to conflicting dependencies for the provided course code. It is defined in [6-capstone-project/pyproject.toml](./6-capstone-project/pyproject.toml)


### Final Projects

1. [Exploratory Data Analysis](./1-exploratory-data-analysis/final-project/IBM-EDA-For-ML-Final-Project.pdf) -> Spotify Audio Features
2. [Regression](./2-regression/final-project/coursera-ibm-regression-final-project.ipynb) -> Movie Popularity
3. [Classification](./3-classification/final-project/coursera_ibm_classification_V01.pdf) -> Mobile Phone Price Classification
4. [Unsupervised Learning](./4-unsupervised-learning/final-project/coursera_ibm_unsupervised_260120.pdf) -> Spotify Audio Features
5. [Deep Learning and Reinforcement Learning](./5-deep-learning-reinforcement-learning/final-project/coursera-ibm-deep-learning-final-project.pdf) -> Phishing Detection
6. [Capstone](./6-capstone-project/final-project/ml-capstone-coursera-v2.pdf) -> Recommender Systems Overview

### Issues

1. Installation of *h5py* with poetry: if it does not work through poetry, try to install with `poetry run pip install h5py==3.14`