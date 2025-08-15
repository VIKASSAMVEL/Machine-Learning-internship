import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import os

# Define dataset path
dataset_path = "e:/codveda/Machine Learning Task List/Task_3_Neural_Networks/Sentiment dataset.csv"

# 1. Load the dataset
try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: Dataset file not found at {dataset_path}")
    exit()

# Assuming the dataset has 'text' and 'sentiment' columns
# Adjust column names if they are different in the actual CSV
if 'Text' not in df.columns or 'Sentiment' not in df.columns:
    print("Error: Expected 'Text' and 'Sentiment' columns not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Clean sentiment labels by stripping whitespace
df['Sentiment'] = df['Sentiment'].str.strip()

# Clean sentiment labels by stripping whitespace
df['Sentiment'] = df['Sentiment'].str.strip()

# Map fine-grained sentiments to broader categories
def map_sentiment(sentiment):
    positive_sentiments = ['Acceptance', 'Accomplishment', 'Admiration', 'Adoration', 'Adrenaline', 'Adventure', 'Affection', 'Amazement', 'Amusement', 'Anticipation', 'Appreciation', 'Arousal', 'ArtisticBurst', 'Awe', 'Blessed', 'Breakthrough', 'Calmness', 'Captivation', 'Celebration', 'Celestial Wonder', 'Challenge', 'Charm', 'Colorful', 'Compassion', 'Compassionate', 'Confidence', 'Confident', 'Connection', 'Contemplation', 'Contentment', 'Coziness', 'Creative Inspiration', 'Creativity', 'Culinary Adventure', 'CulinaryOdyssey', 'Curiosity', 'Dazzle', 'Determination', 'DreamChaser', 'Ecstasy', 'Elation', 'Elegance', 'Empowerment', 'Enchantment', 'Energy', 'Engagement', 'Enjoyment', 'Enthusiasm', 'Euphoria', 'Excitement', 'FestiveJoy', 'Free-spirited', 'Freedom', 'Friendship', 'Fulfillment', 'Grandeur', 'Grateful', 'Gratitude', 'Happiness', 'Happy', 'Harmony', 'Heartwarming', 'Hope', 'Hopeful', 'Hypnotic', 'Iconic', 'Imagination', 'Immersion', 'InnerJourney', 'Inspiration', 'Inspired', 'Joy', 'Joy in Baking', 'JoyfulReunion', 'Kind', 'Kindness', 'Love', 'Marvel', 'Mesmerizing', 'Mindfulness', 'Motivation', "Nature's Beauty", 'Optimism', 'Overjoyed', 'Pensive', 'Playful', 'PlayfulJoy', 'Positive', 'Positivity', 'Pride', 'Proud', 'Radiance', 'Rejuvenation', 'Relief', 'Renewed Effort', 'Resilience', 'Reverence', 'Romance', 'Runway Creativity', 'Satisfaction', 'Serenity', 'Spark', 'Success', 'Surprise', 'Sympathy', 'Tenderness', 'Thrill', 'Thrilling Journey', 'Touched', 'Tranquility', 'Triumph', 'Vibrancy', 'Whimsy', 'Whispers of the Past', 'Winter Magic', 'Wonder', 'Wonderment', 'Yearning', 'Zest']
    negative_sentiments = ['Anger', 'Anxiety', 'Bad', 'Betrayal', 'Bitter', 'Bitterness', 'Boredom', 'Confusion', 'Desolation', 'Despair', 'Desperation', 'Devastated', 'Disappointed', 'Disappointment', 'Disgust', 'Dismissive', 'Envious', 'Envy', 'Exhaustion', 'Fear', 'Fearful', 'Frustrated', 'Frustration', 'Grief', 'Hate', 'Heartache', 'Heartbreak', 'Helplessness', 'Intimidation', 'Isolation', 'Jealous', 'Jealousy', 'Loneliness', 'Loss', 'LostLove', 'Melancholy', 'Miscalculation', 'Negative', 'Numbness', 'Obstacle', 'Overwhelmed', 'Pressure', 'Regret', 'Resentment', 'Ruins', 'Sad', 'Sadness', 'Shame', 'Sorrow', 'Suffering']
    neutral_sentiments = ['Ambivalence', 'Emotion', 'Indifference', 'Intrigue', 'Neutral', 'Nostalgia', 'Reflection', 'Solace', 'Solitude', 'Suspense']

    if sentiment in positive_sentiments:
        return 'Positive'
    elif sentiment in negative_sentiments:
        return 'Negative'
    elif sentiment in neutral_sentiments:
        return 'Neutral'
    else:
        return 'Unknown' # Fallback for any unmapped sentiments

df['Sentiment'] = df['Sentiment'].apply(map_sentiment)

# 2. Preprocess the data
X = df['Text'].astype(str) # Ensure text column is string type
y = df['Sentiment']

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\nOriginal sentiment labels: {label_encoder.classes_}")
print(f"Encoded sentiment labels (0 to {num_classes - 1}): {np.unique(y_encoded)}")

# Split data into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"\nTraining text samples: {X_train_text.shape[0]}")
print(f"Testing text samples: {X_test_text.shape[0]}")

# Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=7500) # Limit features to avoid very large input layer
X_train = tfidf_vectorizer.fit_transform(X_train_text).toarray() # type: ignore
X_test = tfidf_vectorizer.transform(X_test_text).toarray() # type: ignore

print(f"\nShape of X_train after TF-IDF: {X_train.shape}")
print(f"Shape of X_test after TF-IDF: {X_test.shape}")

# 3. Design Neural Network Architecture
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu')) # Added another layer
model.add(Dropout(0.6))

# Output layer
if num_classes == 2:
    model.add(Dense(1, activation='sigmoid')) # Binary classification
    loss_function = 'binary_crossentropy'
else:
    model.add(Dense(num_classes, activation='softmax')) # Multi-class classification
    loss_function = 'sparse_categorical_crossentropy' # Use this if y_encoded is integer labels

model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

model.summary()

# 4. Train the Model
# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=30, # Start with a reasonable number of epochs
                    batch_size=32,
                    validation_split=0.1, # Use a portion of training data for validation
                    callbacks=[early_stopping],
                    verbose=1)


print("\nModel training complete.")

# 5. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Visualize training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Save the plots
plot_path = "e:/codveda/Machine Learning Task List/Task_3_Neural_Networks/training_history.png"
plt.tight_layout()
plt.savefig(plot_path)
print(f"\nTraining history plot saved as '{plot_path}'")

print("\nNeural network model built and evaluated successfully.")
