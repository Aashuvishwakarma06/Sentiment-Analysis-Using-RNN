import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load model
model = load_model("model/sentiment_rnn.h5")

# Load IMDb word index
word_index = imdb.get_word_index()

# ye wala isliye  error de rahakyuki ye uper ka 5000 row per hi trained h
# Encode function
# def encode_review(text):
#     words = text.lower().split()
#     encoded = []

#     for word in words:
#         index = word_index.get(word, 2) + 3
#         encoded.append(index)

#     return encoded
# ye wala isliye  error de rahakyuki ye uper ka 5000 row per hi trained h




def encode_review(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        index = word_index.get(word, 2) + 3

        # Keep only words inside training vocabulary
        if index < 5000:
            encoded.append(index)
        else:
            encoded.append(2)   # unknown token

    return encoded
# UI
st.title("🎬 Sentiment Analysis using RNN")
st.write("Enter a movie review and predict sentiment")

review = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review:
        encoded_review = encode_review(review)
        padded_review = pad_sequences([encoded_review], maxlen=100)

        prediction = model.predict(padded_review)

        if prediction[0][0] > 0.5:
            st.success("Positive Sentiment 😊")
            st.write(f"Confidence Score: {prediction[0][0]:.2f}")
        else:
            st.error("Negative Sentiment 😞")
            st.write(f"Confidence Score: {prediction[0][0]:.2f}")
    else:
        st.warning("Please enter a review first.")