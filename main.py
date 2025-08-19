# main.py
import numpy as np
from models.gpt import NanoGPT

if __name__ == "__main__":
    # Define model parameters
    VOCAB_SIZE = 1000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    MAX_LEN = 50

    # Initialize the model
    gpt_model = NanoGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    )

    # Create dummy input data
    # (batch_size, sequence_length)
    batch_size = 2
    sequence_length = 10
    dummy_input = np.random.randint(0, VOCAB_SIZE, (batch_size, sequence_length))

    # Run a forward pass
    output_logits = gpt_model.forward(dummy_input)

    # Print the output shape to verify
    print("Shape of input:", dummy_input.shape)
    print("Shape of output logits:", output_logits.shape)