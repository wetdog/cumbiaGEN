import tensorflow as tf
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer

# Global parameters
BATCH_SIZE = 32
DATA_PATH = "/content/cumbiaGEN/Code/dataset.json"
N_MELODIES=10

transformer_model = Transformer(
    num_layers=2,
    d_model=64,
    num_heads=2,
    d_feedforward=128,
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
    max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
    dropout_rate=0.1,
)



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
    with open(f"/content/melody_{i}.txt", "w") as f:
        f.write(new_melody) 