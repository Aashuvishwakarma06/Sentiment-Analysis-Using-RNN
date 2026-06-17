import pickle
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# Pad sequences (same length reviews)
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

# Save processed files
with open("dataset/processed/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)

with open("dataset/processed/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

with open("dataset/processed/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("dataset/processed/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Data preprocessing completed successfully!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)