# Movie Review Sentiment Analysis using RNN

This project demonstrates sentiment analysis of movie reviews using Recurrent Neural Networks (RNNs), specifically SimpleRNN and LSTM layers, implemented with Keras and TensorFlow.

## Project Description

The goal of this project is to build a model that can classify movie reviews as either positive or negative. The dataset consists of short movie review titles, pre-processed and tokenized for use in a neural network. The project covers data loading, pre-processing, model building, training, evaluation, and prediction.

## Files

* `negative.txt`: Contains negative movie review titles.
* `positive.txt`: Contains positive movie review titles.
* `movie_review_rnn.ipynb`: Jupyter Notebook containing the code for the project.
* `README.md`: This file provides project documentation.

## Libraries Used

* `matplotlib.pyplot`: For plotting graphs.
* `seaborn`: For enhanced data visualization.
* `numpy`: For numerical operations.
* `re`: For regular expression operations.
* `statistics`: For statistical calculations.
* `tensorflow` and `keras`: For building and training neural networks.
* `sklearn`: For data splitting and evaluation metrics.

## Data Loading and Pre-processing

1.  **Load Data:** The `load_data` function reads the review titles from the text files, filters out non-alphanumeric characters, and converts them to lowercase.
2.  **Combine and Label Data:** Positive and negative reviews are combined into a single list, and labels are created.
3.  **Tokenization:** The `Tokenizer` from Keras is used to convert the reviews into sequences of integers.
4.  **Padding:** The sequences are padded to ensure all reviews have the same length.
5.  **Data Splitting:** The data is split into training and test sets using `train_test_split` from sklearn.
6.  **Label Encoding:** The labels are converted to one-hot encoded vectors.

## Model Architecture

Two models are built:

1.  **Model 1 (Simple Embedding and Dense Layers):**
    * Embedding layer to convert tokenized reviews to dense vectors.
    * Flatten layer to convert the 2D output of the Embedding layer to 1D.
    * Dense layer with softmax activation for classification.

2.  **Model 2 (RNN with LSTM):**
    * Embedding layer.
    * SimpleRNN layers
    * Dropout layer
    * LSTM Layer
    * Dense layer with sigmoid activation for classification.

## Model Training and Evaluation

* Both models are compiled with the `rmsprop` optimizer and `categorical_crossentropy` loss.
* The models are trained using the training data, and their performance is evaluated on the test data.
* The `assess_model` function calculates and displays precision, recall, F1-score, and the confusion matrix.
* The `plot` function creates a plot of the training and validation accuracy and loss.

## Prediction

* New review titles are tokenized, padded, and passed to the trained models for sentiment prediction.
* The `to_word_label` function is used to convert the numerical predictions to sentiment labels (positive or negative).

## How to Run

1.  Ensure you have all the required libraries installed.
2.  Place the `negative.txt` and `positive.txt` files in the same directory as the Jupyter Notebook.
3.  Open and run the `movie_review_rnn.ipynb` notebook.

## Results

The models achieved an accuracy of around 50% on the test data. This indicates that the models are not performing well on this dataset. Further tuning and exploration of other model architectures might be needed to improve performance. The models have a difficult time finding patterns in the very short review titles.

## Future Improvements

* Experiment with different RNN architectures, such as GRUs.
* Tune hyperparameters to improve model performance.
* Use a larger and more diverse dataset.
* Explore different tokenization and pre-processing techniques.
* Implement word embeddings like GloVe or Word2Vec.
* Add regularization to improve the model.
