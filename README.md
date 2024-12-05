# Disaster Tweet Classification - Kaggle Competition

This project focuses on classifying tweets as disaster-related or non-disaster-related using a machine learning model. The dataset used for this project is sourced from the Kaggle competition: [NLP - Getting Started](https://www.kaggle.com/c/nlp-getting-started/code).

## Overview

The goal of this project is to predict whether a tweet is about a real disaster or not. To achieve this, we explore the dataset, preprocess the text data, and apply a neural network model for classification. The project was completed using a starter notebook from Kaggle and further customized to meet the project's specific needs.

## Dataset

The dataset consists of two main CSV files:
- `train.csv`: Contains training data, including the text of the tweet and the target label (1 for disaster, 0 for non-disaster).
- `test.csv`: Contains tweet text for testing, without the target label.

The data includes additional metadata such as tweet `id`, `keyword`, and `location`. The first step involves loading the dataset and evaluating its structure and memory usage.

### Dataset Analysis

After loading the data:
- **Training Set**: The dataset consists of 7,613 training samples, each having 5 columns (ID, keyword, location, tweet text, and target label).
- **Test Set**: Contains 3,263 samples with 4 columns (ID, keyword, location, and tweet text).

The length of each tweet was calculated and added as a new feature. Descriptive statistics for tweet lengths were computed, showing that tweet lengths vary between 7 and 157 characters in the training set.

## Data Preprocessing

### Splitting the Dataset

The dataset was split into training and validation sets using an 80-20 split. This was done to train the model on one portion of the data and validate it on another to assess its performance.

### Text Preprocessing

The tweet texts were processed using the DistilBERT tokenizer. This tokenizer handles the conversion of words into a format that the BERT-based model can understand. The text data was padded and truncated to a fixed length (160 tokens).

### Data Augmentation

For training efficiency, the text data was converted into TensorFlow datasets, enabling batch processing. The data was shuffled, and batches of 32 samples were created for both training and validation datasets. Additionally, prefetching was implemented for optimized input pipeline processing.

## Model Development

The project uses a **DistilBERT** model, which is a smaller, faster version of the BERT (Bidirectional Encoder Representations from Transformers) model. DistilBERT is particularly well-suited for NLP tasks like this one because of its ability to understand the contextual meaning of words in a sentence.

### Model Configuration

The model was configured as follows:
- **Preprocessor**: The preprocessor transforms input text into the format expected by DistilBERT, including tokenization and padding.
- **Classifier**: The classifier layer consists of a DistilBERT backbone followed by a dense output layer with a softmax activation to classify tweets into two categories (disaster or non-disaster).

### Compilation

The model was compiled using the Adam optimizer with a learning rate of \(5 \times 10^{-5}\). The loss function used was Sparse Categorical Crossentropy, which is suitable for multi-class classification problems like this one. Accuracy was tracked during training to evaluate the model’s performance.

## Model Training

The model was trained for 10 epochs, but only a small subset of the data was used for training due to time constraints. Each epoch was evaluated for both training and validation accuracy.

### Results

Training the model yielded promising results, with validation accuracy gradually improving over several epochs. The model was able to achieve a high accuracy during training and validation steps, indicating that the model was learning effectively.

---

## Conclusion

This project demonstrates how to preprocess and classify tweet data using state-of-the-art transformer models like DistilBERT. Although only a small subset of the data was used for training, the results show that the model is capable of distinguishing between disaster-related and non-disaster-related tweets. Further optimization and model training can yield even better results.
