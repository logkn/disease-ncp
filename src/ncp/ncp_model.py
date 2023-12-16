from ncp_config import (
    DEFAULT_NCP_TRAINING_CONFIG,
    EncoderDecoderConfig,
)

import torch
from torch.utils.data import DataLoader, TensorDataset
from ncps.wirings import AutoNCP
from ncps.torch import CfC
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import platform
import sys
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim, n_units):
        super(Encoder, self).__init__()
        # Define AutoNCP wiring for the encoder
        encoder_wiring = AutoNCP(n_units, latent_dim)
        self.rnn = CfC(input_size, encoder_wiring)

    def forward(self, x, dt):
        _, hidden_state = self.rnn(x, dt)
        return hidden_state


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_units):
        super(Decoder, self).__init__()
        # Define AutoNCP wiring for the decoder
        decoder_wiring = AutoNCP(n_units, output_size)
        self.rnn = CfC(input_size, decoder_wiring)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_state, dt):
        decoder_output, _ = self.rnn(dt, initial_state=hidden_state)
        prediction = self.output_layer(decoder_output)
        return prediction


class EncoderDecoder:
    def __init__(self, config: EncoderDecoderConfig):
        self.latent_dim = config["latent_dim"]

        input_size = 1
        self.latent_dim = config["latent_dim"]
        output_size = 1

        n_units_encoder = config["encoder_n_units"]
        n_units_decoder = config["decoder_n_units"]

        self.encoder = Encoder(
            input_size, self.latent_dim, n_units_encoder
        )  # Adjust sizes as needed
        self.decoder = Decoder(
            1, self.latent_dim, output_size, n_units_decoder
        )  # Assuming dt is 1D

    def forward(self, x_encoder, dt_encoder, dt_decoder):
        hidden_state = self.encoder(x_encoder, dt_encoder)
        prediction = self.decoder(hidden_state, dt_decoder)
        return prediction


if __name__ == "__main__":
    enc_dec = EncoderDecoder(DEFAULT_NCP_TRAINING_CONFIG["model_config"])
    # ncp.visualize()
    # ncp.train()
