# Sentiment Analysis of Movie Reviews

## Overview
This project includes two main parts: Data Science (DS) and Machine Learning Engineering (MLE). The DS part focuses on exploring the dataset, feature engineering, model selection, and performance evaluation, while the MLE part deals with containerizing the training and inference processes.

## Table of Contents
- [DS Part](#ds-part)
  - [Conclusions from EDA](#conclusions-from-eda)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection)
  - [Performance Evaluation](#performance-evaluation)
  - [Business Applications](#business-applications)
  - [Conclusion](#conclusion)
- [MLE Part](#mle-part)
  - [Project Structure](#project-structure)
  - [Additional steps](#additional-steps)
  - [How to Run](#how-to-run)
    - [Training](#training)
    - [Inference](#inference)

## DS Part

### Conclusions from EDA
Through exploratory data analysis, key characteristics of the dataset were identified:
- **Class Distribution**: The dataset is balanced, featuring an equal number of positive and negative reviews.
- **Review Length**: The review lengths vary significantly, with a distribution that shows most reviews are under 2000 characters but some extend much longer.

### Feature Engineering
Feature engineering process included several steps to prepare the text data for modeling:
- **Text Preprocessing**: Implemented tokenization, removal of stop words, and punctuation elimination, which are essential for reducing noise.
- **Stemming vs. Lemmatization**: Both techniques were compared and **Stemming** was opted, finding it provided a better foundation and accuracy for vectorization approach.
- **Vectorization**: TF-IDF and Count Vectorization were evaluated, choosing **TF-IDF** for its ability to highlight words that are particularly indicative of sentiment in individual reviews (and in terms of slightly better accuracy).

### Model Selection
Various models and their configurations were explored to identify the most effective approach:
- **Evaluated Models**: Logistic Regression, SVC, LinearSVC, and Random Forest.
- **Hyperparameter Tuning**: Conducted using GridSearchCV, this process was essential in enhancing the models' accuracy.
- **Final Choice**: **LinearSVC** was selected based on its superior performance, achieving an accuracy of around **90%**.

### Performance Evaluation
The final model demonstrated excellent performance on the test dataset:
- **Accuracy**: Achieved an accuracy of around **90%**.

### Business Applications
The developed sentiment analysis model holds significant potential for various business applications:
- **Automated Review Classification**: Enables businesses to automatically sort customer feedback across platforms, essentially without human intervention.
- **Enhanced Customer Insights**: Facilitates sentiment analysis, allowing businesses to swiftly address customer concerns and improve satisfaction.
- **Market Research**: Offers valuable insights into consumer preferences and market trends, aiding strategic decision-making.

### Conclusion
This project underscores the effectiveness of combining NLP and machine learning to interpret and classify sentiments in text data. The model not only meets the set accuracy goals but also opens avenues for practical applications in customer feedback analysis and beyond.

## MLE Part

### Project Structure
The MLE part consists of the following structure:

```
MLE part/
|-- data/
| |-- preprocessed_test/
| |-- raw/
| | |-- test.csv
| | |-- train.csv
|-- outputs/
| |-- models/
| |-- predictions/
|-- src/
| |-- inference/
| | |-- Dockerfile
| | |-- requirements.txt # Specific to inference
| | |-- run_inference.py
| |-- train/
| | |-- Dockerfile
| | |-- requirements.txt # Specific to training
| | |-- train.py
|-- .gitignore
```

### Additional steps
As the data folder should be git ignored according to project requirements, please follow next step:
- Create folder data and other subfolders according to the project structure above

### How to Run

#### Training
To train the sentiment analysis model:
1. Navigate to project's root folder (MLE part) `MLE part/` directory.
2. Build the Docker image using the provided Dockerfile:
    ```
    docker build -f src/train/Dockerfile -t sentiment-analysis-training .
    ```
3. Run the Docker container:
    ```
    docker run -v "path_to_MLE part/data:/usr/src/app/data" -v "path_to_MLE part/outputs:/usr/src/app/outputs" sentiment-analysis-training
    ```

#### Inference
To perform inference using the trained model:
1. Navigate to project's root folder (MLE part) `MLE part/` directory.
2. Build the Docker image using the provided Dockerfile:
    ```
    docker build -f src/inference/Dockerfile -t sentiment-analysis-inference .
    ```
3. Run the Docker container:
    ```
    docker run -v "path_to_MLE part/data:/usr/src/app/data" -v "path_to_MLE part/outputs:/usr/src/app/outputs" sentiment-analysis-inference
    ```