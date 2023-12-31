"""
This file contains the training pipeline for a Transformer model specialized in
melody generation. It includes functions to calculate loss, perform training steps,
and orchestrate the training process over multiple epochs. The script also
demonstrates the use of the MelodyGenerator class to generate a melody after training.

The training process uses a custom implementation of the Transformer model,
defined in the 'transformer.py' module, and prepares data using the
MelodyPreprocessor class from 'melodypreprocessor.py'.

Global parameters such as the number of epochs, batch size, and path to the dataset
are defined. The script supports dynamic padding of sequences and employs the
Sparse Categorical Crossentropy loss function for model training.

For simplicity's sake training does not deal with masking of padded values
in the encoder and decoder. Also, look-ahead masking is not implemented.
Both of these are left as an exercise for the student.

Key Functions:
- _calculate_loss_function: Computes the loss between actual and predicted sequences.
- _train_step: Executes a single training step, including forward pass and backpropagation.
- train: Runs the training loop over the entire dataset for a given number of epochs.
- _right_pad_sequence_once: Utility function for padding sequences.

The script concludes by instantiating the Transformer model, conducting the training,
and generating a sample melody using the trained model.
"""

import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer
import argparse
from argparse import RawTextHelpFormatter
import os
import matplotlib.pyplot as plt

# Loss function and optimizer
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = Adam()


def train(train_dataset, transformer, epochs):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.

    Parameters:
        train_dataset (tf.data.Dataset): The training dataset.
        transformer (Transformer): The Transformer model instance.
        epochs (int): The number of epochs to train the model.
    """
    print("Training the model...")
    models_path = "/content/models"
    losses = []

    if not(os.path.exists(models_path)):
        os.mkdir(models_path)
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over each batch in the training dataset
        for (batch, (input, target)) in enumerate(train_dataset):
            # Perform a single training step
            batch_loss = _train_step(input, target, transformer)
            total_loss += batch_loss
            
            print(
                f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy()}"
            )
        losses.append(total_loss)
            
        transformer.save_weights(os.path.join(models_path,f"epoch_{epoch}/"), save_format="tf")
    
    # plot loss at the end
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"training_curves.png")
    



@tf.function
def _train_step(input, target, transformer):
    """
    Performs a single training step for the Transformer model.

    Parameters:
        input (tf.Tensor): The input sequences.
        target (tf.Tensor): The target sequences.
        transformer (Transformer): The Transformer model instance.

    Returns:
        tf.Tensor: The loss value for the training step.
    """
    # Prepare the target input and real output for the decoder
    # Pad the sequences on the right by one position
    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Forward pass through the transformer model
        # TODO: Add padding mask for encoder + decoder and look-ahead mask
        # for decoder
        predictions = transformer(input, target_input, True, None, None, None)

        # Compute loss between the real output and the predictions
        loss = _calculate_loss(target_real, predictions)

    # Calculate gradients with respect to the model's trainable variables
    gradients = tape.gradient(loss, transformer.trainable_variables)

    # Apply gradients to update the model's parameters
    gradient_variable_pairs = zip(gradients, transformer.trainable_variables)
    optimizer.apply_gradients(gradient_variable_pairs)

    # Return the computed loss for this training step
    return loss


def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.

    Parameters:
        real (tf.Tensor): The actual target sequences.
        pred (tf.Tensor): The predicted sequences by the model.

    Returns:
        average_loss (tf.Tensor): The computed loss value.
    """

    # Compute loss using the Sparse Categorical Crossentropy
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Create a mask to filter out zeros (padded values) in the real sequences
    boolean_mask = tf.math.equal(real, 0)
    mask = tf.math.logical_not(boolean_mask)

    # Convert mask to the same dtype as the loss for multiplication
    mask = tf.cast(mask, dtype=loss_.dtype)

    # Apply the mask to the loss, ignoring losses on padded positions
    loss_ *= mask

    # Calculate average loss, excluding the padded positions
    total_loss = tf.reduce_sum(loss_)
    number_of_non_padded_elements = tf.reduce_sum(mask)
    average_loss = total_loss / number_of_non_padded_elements

    return average_loss


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end.

    Parameters:
        sequence (tf.Tensor): The sequence to be padded.

    Returns:
        tf.Tensor: The padded sequence.
    """
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        description="""Resample a folder recusively with librosa
                       Can be used in place or create a copy of the folder as an output.\n\n
                       Example run:
                            python TTS/bin/resample.py
                                --input_dir /root/LJSpeech-1.1/
                                --output_sr 22050
                                --output_dir /root/resampled_LJSpeech-1.1/
                                --file_ext wav
                                --n_jobs 24
                    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="Path to the file of preprocessed dataset",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path where the melodies are stored",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="training batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        required=False,
        help="training batch size",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        required=False,
        help="temperature for generation",
    )

    parser.add_argument(
        "--positions",
        type=int,
        default=100,
        required=False,
        help="MAX_POSITIONS_IN_POSITIONAL_ENCODING",
    )


    args = parser.parse_args()
    # Global parameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_path
    MAX_POSITIONS_IN_POSITIONAL_ENCODING = args.positions
    TEMPERATURE=args.temperature

    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    print("vocab_size", vocab_size)

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

    train(train_dataset, transformer_model, EPOCHS)
    
    transformer_model.save_weights("/content/weigths/", save_format="tf")

    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer,temperature=TEMPERATURE,max_length=25
    )

    # create dir for saving melodies

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    start_sequences = [["r-0.5", "G3-0.5", "C4-0.5", "D4-0.5", "E4-0.5"],
                        ["r-0.5", "C4-0.5", "E4-0.5", "G4-0.5", "F4-1.0"],
                        ["r-1.0", "A4-1.0", "E4-1.0", "A4-1.0", "E4-1.0"],
                        ["r-2.0", "r-1.0", "D4-0.5", "F4-0.5", "E4-0.5", "D4-0.5"],
                        ["r-2.0", "r-1.0", "D4-0.5","D4-1.0", "F4-1.0", "A4-1.0"],
                        ["r-1.0", "A4-1.0", "E4-1.0", "A4-1.0", "r-0.5"],
                        ["r-0.5", "G3-0.5", "C4-0.5", "D4-0.5", "E4-0.5"],
                        ["r-0.5", "C4-0.5", "E4-0.5", "G4-0.5", "F4-1.0"],
                        ["r-1.0", "A4-1.0", "E4-1.0", "A4-1.0", "E4-1.0"],
                        ["r-2.0", "r-1.0", "D4-0.5", "F4-0.5", "E4-0.5", "D4-0.5"],
                        ["r-2.0", "r-1.0", "D4-0.5","D4-1.0", "F4-1.0", "A4-1.0"],
                        ["r-1.0", "A4-1.0", "E4-1.0", "A4-1.0", "r-0.5"]]
    
    for i, start_sequence in enumerate(start_sequences):
        
        new_melody = melody_generator.generate(start_sequence)
        print(f"Generated melody: {new_melody}")

        # save melody
        with open(os.path.join(OUTPUT_DIR, f"melody_{i}.txt"), "w") as f:
            f.write(new_melody)