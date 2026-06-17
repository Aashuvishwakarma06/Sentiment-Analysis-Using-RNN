import pickle
from tensorflow.keras.models import load_model

# Load test data
with open("dataset/processed/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("dataset/processed/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load trained model
model = load_model("model/sentiment_rnn.h5")

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)