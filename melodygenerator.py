import tensorflow as tf
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21

class MelodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model_path =model_path
        self.model = tf.keras.models.load_model(model_path)

        with open(MAPPING_PATH,"r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        #create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to the max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = tf.keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            # add another dimention for tf.keras expects a batch of samples
            # It becomes (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            #update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        # temp -> infinity super random
        #temp -> 0 super rigid
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions)) #softmax

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody,step_duration=0.25, format="midi", file_name="mel.midi"):

        # create a music21 stream
        stream = m21.stream.Stream()

        #parse all the symbols in the melody and create note/rest objects
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            #handle case in which we have a note/rest
            if symbol != "_":
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter

                    # handle rest
                    if start_symbol == "r" or i + 1 == len(melody):
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    #reset the step counter
                    step_counter = 1

                start_symbol = symbol

            else:
                step_counter += 1


        #write the m21 stream to a midi file
        stream.write(format, file_name)

if __name__ == "__main__":
    mg = MelodyGenerator()
    seed ="67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed2, 1000, SEQUENCE_LENGTH, 0.6)
    print(melody)
    mg.save_melody(melody)
