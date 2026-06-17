from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

def build_model():
    model = Sequential()

    model.add(Embedding(
        input_dim=5000,
        output_dim=64,
        input_shape=(100,)
    ))

    model.add(SimpleRNN(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.build(input_shape=(None, 100))
    model.summary()



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# def build_model():
#     model = Sequential()

#     # Convert word IDs into dense vectors
#     model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))

#     # RNN layer
#     model.add(SimpleRNN(64))

#     # Output layer
#     model.add(Dense(1, activation='sigmoid'))

#     # Compile model
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     return model


# if __name__ == "__main__":
#     model = build_model()
#     model.summary()