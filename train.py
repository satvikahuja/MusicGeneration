from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import tensorflow as tf

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LEARNING_RATE = 0.001
LOSS = "sparse_categorical_crossentropy"
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def build_model(output_units, num_units, loss, learning_rate):

    # create model architecture
    input = tf.keras.layers.Input(shape=(None, output_units))
    x = tf.keras.layers.LSTM(num_units[0])(input)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(output_units, activation = "softmax")(x)

    model = tf.keras.Model(input, output)

    #compile model

    model.compile(loss=loss,
                  optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics = ["accuracy"])
    model.summary()

    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
