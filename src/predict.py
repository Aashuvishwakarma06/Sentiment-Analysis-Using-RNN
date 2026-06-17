from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load model
model = load_model("model/sentiment_rnn.h5")

# Load word index
word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        index = word_index.get(word, 2) + 3
        encoded.append(index)

    return encoded


review = input("Enter your review: ")

encoded_review = encode_review(review)
padded_review = pad_sequences([encoded_review], maxlen=100)

prediction = model.predict(padded_review)

if prediction[0][0] > 0.5:
    print("Sentiment: Positive 😊")
else:
    print("Sentiment: Negative 😞")