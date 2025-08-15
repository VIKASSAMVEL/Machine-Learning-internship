# Task 3: Neural Networks Model Report

## 1. Task Description
This task involved building a simple feed-forward neural network using TensorFlow/Keras for a text classification task (sentiment analysis). The objectives were to load and preprocess a sentiment dataset, design a neural network architecture, train the model using backpropagation, and evaluate the model using accuracy and visualize the training/validation loss.

## 2. Datasets Used
The dataset used for this task was:
- `Sentiment dataset.csv`

This dataset contains text data and corresponding sentiment labels.

## 3. Model Implemented
A Feed-Forward Neural Network using TensorFlow/Keras was implemented.

## 4. Features and Target
- **Target Variable (y):** `Sentiment` (categorical: 'Positive', 'Negative', 'Neutral')
- **Features (X):** `Text` column from the dataset.

## 5. Preprocessing Steps
1.  **Loading Data:** The dataset was loaded using pandas `read_csv`.
2.  **Sentiment Label Cleaning and Mapping:**
    - Whitespace was stripped from sentiment labels.
    - Fine-grained sentiment labels were mapped to broader categories: 'Positive', 'Negative', or 'Neutral'.
3.  **Target Variable Encoding:** The `Sentiment` (target variable) was encoded into numerical labels using `LabelEncoder`.
4.  **Data Splitting:** The data was split into training and testing sets (80% training, 20% testing).
5.  **Text Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) vectorization was applied to the `Text` feature using `TfidfVectorizer` to convert text data into numerical feature vectors. `max_features` was set to 7500.

## 6. Model Architecture
The neural network was designed as a Sequential model with the following layers:
-   **Input Layer:** A Dense layer with 512 units, 'relu' activation, and input shape determined by the TF-IDF vectorizer's output (`X_train.shape[1]`).
-   **Hidden Layers:**
    -   A Dropout layer with a rate of 0.6.
    -   A Dense layer with 256 units and 'relu' activation.
    -   A Dropout layer with a rate of 0.6.
    -   A Dense layer with 128 units and 'relu' activation.
    -   A Dropout layer with a rate of 0.6.
-   **Output Layer:**
    -   If the number of unique sentiment classes is 2 (binary classification), a Dense layer with 1 unit and 'sigmoid' activation.
    -   If the number of unique sentiment classes is greater than 2 (multi-class classification), a Dense layer with `num_classes` units and 'softmax' activation.

The model was compiled with the 'adam' optimizer. The loss function was 'binary_crossentropy' for binary classification and 'sparse_categorical_crossentropy' for multi-class classification. 'accuracy' was used as the evaluation metric.

## 7. Model Training and Evaluation

The model was trained with the following parameters:
-   **Epochs:** 30 (with EarlyStopping)
-   **Batch Size:** 32
-   **Validation Split:** 0.1 (10% of training data used for validation)
-   **Callbacks:** EarlyStopping (monitoring 'val_loss' with patience of 7 epochs, restoring best weights).

**Evaluation Metrics on Test Set:**
-   **Test Loss:** 0.5689
-   **Test Accuracy:** 0.8844

## 8. Interpretation of Results

The neural network model achieved a test accuracy of 0.8844, indicating that it correctly classified approximately 88.44% of the sentiment labels in the test set. The test loss of 0.5689 suggests the model's prediction error on unseen data. The training history plot (saved as `training_history.png`) would provide further insights into the model's learning process, showing how accuracy and loss evolved over epochs for both training and validation sets, and if overfitting occurred. The use of EarlyStopping helped in preventing overfitting by monitoring validation loss and restoring the best weights.

## 9. Script Execution and Output

Below is the console output from the execution of `neural_network_model.py`.

```
Dataset loaded successfully. First 5 rows:
   Unnamed: 0.1  Unnamed: 0                                               Text    Sentiment  ...  Year Month Day Hour
0             0           0   Enjoying a beautiful day at the park!        ...   Positive    ...  2023     1  15   12
1             1           1   Traffic was terrible this morning.           ...   Negative    ...  2023     1  15    8
2             2           2   Just finished an amazing workout! 💪          ...   Positive    ...  2023     1  15   15
3             3           3   Excited about the upcoming weekend getaway!  ...   Positive    ...  2023     1  15   18
4             4           4   Trying out a new recipe for dinner tonight.  ...   Neutral     ...  2023     1  15   19

[5 rows x 15 columns]

Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 732 entries, 0 to 731
Data columns (total 15 columns):
 #   Column        Non-Null Count  Dtype
---  ------
 0   Unnamed: 0.1  732 non-null    int64
 1   Unnamed: 0    732 non-null    int64
 2   Text          732 non-null    object
 3   Sentiment     732 non-null    object
 4   Timestamp     732 non-null    object
 5   User          732 non-null    object
 6   Platform      732 non-null    object
 7   Hashtags      732 non-null    object
 8   Retweets      732 non-null    float64
 9   Likes         732 non-null    float64
 10  Country       732 non-null    object
 11  Year          732 non-null    int64
 12  Month         732 non-null    int64
 13  Day           732 non-null    int64
 14  Hour          732 non-null    int64
dtypes: float64(2), int64(6), object(7)
memory usage: 85.9+ KB

Original sentiment labels: [\'Negative\' \'Neutral\' \'Positive\' \'Unknown\]
Encoded sentiment labels (0 to 3): [0 1 2 3]

Training text samples: 585
Testing text samples: 147

Shape of X_train after TF-IDF: (585, 2240)
Shape of X_test after TF-IDF: (147, 2240)
Model: "sequential"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━
  dense (Dense)                        │ (None, 512)                 │       1,147,392 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  dropout (Dropout)                    │ (None, 512)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  dense_1 (Dense)                      │ (None, 256)                 │         131,328 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  dropout_1 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  dense_2 (Dense)                      │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  dropout_2 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  dense_3 (Dense)                      │ (None, 4)                   │             516 │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━
 Total params: 1,312,132 (5.01 MB)
 Trainable params: 1,312,132 (5.01 MB)
 Non-trainable params: 0 (0.00 B)

Test Loss: 0.5689
Test Accuracy: 0.8844

Training history plot saved as 'e:/codveda/Machine Learning Task List/Task_3_Neural_Networks/training_history.png'
```
