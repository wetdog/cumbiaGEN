import tensorflow as tf
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor

# Global parameters
BATCH_SIZE = 32
DATA_PATH = "/content/cumbiaGEN/Code/dataset.json"
N_MELODIES=10

transformer_model = tf.keras.models.load_model('/content/model.h5py')

melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)


for i in range(N_MELODIES):
    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer
    )
    start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")

    # save melody
    with open("/content/melody_.txt", "w") as f:
        f.write(new_melody) 