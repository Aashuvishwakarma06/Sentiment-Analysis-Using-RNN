import pickle
from model_building import build_model

# Load processed data
with open("dataset/processed/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("dataset/processed/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

# Build model
model = build_model()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Save model
model.save("model/sentiment_rnn.h5")

# Save training history
with open("model/training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Training completed successfully!")
print("Model saved successfully!")