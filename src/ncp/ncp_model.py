from ncp_config import EncoderDecoderConfig, DecoderNCPConfig, EncoderNCPConfig, NCPTrainingConfig

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

# TODO
# This model will be an encoder-decoder RNN
# The encoder will specialize in taking the first N time units of the observations and encoding them into a latent space
# (i.e. its hidden state)
# The decoder will take the latent space, in addition to the time to the next observation, and predict the next observation
# The loss function will be the MSE between the predicted observations and the actual observations, with
# exponential decay on the MSE for observations further in the future


# wiring = AutoNCP(16, out_features)  # 16 units, 1 motor neuron

# ltc_model = LTC(in_features, wiring, batch_first=True)
# learn = SequenceLearner(ltc_model, lr=0.01)
# trainer = pl.Trainer(
#     logger=pl.loggers.CSVLogger("log"),
#     max_epochs=400,
#     gradient_clip_val=1,  # Clip gradient to stabilize training
#     gpus=0,
# )


# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    # TODO:
    # - implement `self.load_data`
    # - implement visualizations (maybe move these to trainer)
    
    def __init__(self, model, chains, lr: float = 0.005, batch_size: int = 32, validation_percent: float = 0.15):
        super().__init__()
        self.model = model
        self.lr = lr
        # self.data_path = data_path
        # self.batch_size = batch_size
        self.chains = chains
        self.validation_percent = validation_percent
        
        self.load_data()

    def load_data(self):
        # TODO: this function should make training and validation
        # datasets from the chains
        raise NotImplementedError()


    # def load_data_chain(self, chain_path):

    #     data = pd.read_csv(chain_path)

    #     N = len(data)

    #     self.batch_size = self.batch_size - (N % self.batch_size)

    #     self.data_t = data["time"].values.astype(np.float32)
    #     self.data_x = data["emission"].values.astype(np.float32)
    #     self.data_state = data["state"].values.astype(np.float32)

    #     n_batches = N // self.batch_size

    #     print(n_batches)

    #     batched_t = torch.Tensor(self.data_t.reshape([n_batches, -1, 1]))
    #     batched_x = torch.Tensor(self.data_x.reshape([n_batches, -1, 1]))

    #     train_number = int((1 - self.validation_percent) * len(batched_t))

    #     print("train_number", train_number)
    #     print("validation_number", len(batched_t) - train_number)

    #     batched_t_train = batched_t[:train_number]
    #     batched_x_train = batched_x[:train_number]

    #     batched_t_val = batched_t[train_number:]
    #     batched_x_val = batched_x[train_number:]

    #     self.train_dataloader = DataLoader(
    #         TensorDataset(batched_t_train, batched_x_train),
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=4,
    #     )

    #     self.val_dataloader = DataLoader(
    #         TensorDataset(batched_t_val, batched_x_val),
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=4,
    #     )


    def visualize_data(self, first_n: int = 100):
        plt.figure(figsize=(6, 4))
        cumulative_times = np.cumsum(self.data_t[:first_n])
        colors = plt.cm.viridis(self.data_state[:first_n].astype(int) / float(max(self.data_state[:first_n].astype(int))))

        plt.scatter(cumulative_times, self.data_x[:first_n], c=colors, s=10)
        plt.plot(cumulative_times, self.data_x[:first_n], c='black', linewidth=0.5)

        plt.title("Before training")
        plt.xlabel("Time")
        plt.ylabel("Emission")
        plt.show()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class EncoderNCP(nn.Module):
    def __init__(self, encoder_config:EncoderNCPConfig, latent_dim:int):
        self.config = encoder_config
        self.units = self.config["n_units"]
        self.wiring = AutoNCP(
            self.units,
            latent_dim
        )
        self.rnn = CfC(
            2, self.wiring, return_sequences=False
        )

    def forward(self, x, hidden):
        return self.rnn(x, hidden)

class DecoderNCP(nn.Module):
    def __init__(self, decoder_config:DecoderNCPConfig, latent_dim:int):
        self.config = decoder_config
        self.units = self.config["n_units"]
        self.wiring = AutoNCP(
            self.units,
            latent_dim
        )
        self.rnn = CfC(
            1, self.wiring, return_sequences=True
        )
        self.out = nn.Linear(
            latent_dim,
            1,
        )

    def forward(self, elapsed_times, encoder_hidden):
        rnn_out, rnn_hidden = self.rnn(elapsed_times, encoder_hidden)
        transformed_outs = self.out(rnn_out)
        return rnn_out, rnn_hidden

class EncoderDecoderNCP:
    def __init__(self, config:EncoderDecoderConfig):
        self.latent_dim = config["latent_dim"]
        self.encoder = EncoderNCP(config["encoder_config"], latent_dim)
        self.decoder = DecoderNCP(config["decoder_config"], latent_dim)


    def forward(self, input_sequence, prediction_times, hidden):
        encoder_out, encoder_hidden = self.encoder(input_sequence)
        decoder_outs, decoder_hidden = self.decoder(prediction_times, encoder_out)
        return decoder_outs

class NCPTrainer:

    def __init__(self, training_config:NCPTrainingConfig):
        self.model_config = training_config[""]
        self.model = EncoderDecoderNCP(model_config)
        self.training_config = training_config
        self.lr = training_config["learning_rate"]
        self.data_path = training_config["data_path"]
        self.sigma_lvl = training_config["sigma_lvl"]
        
        self.chains = self._get_chains(self)
        self.jump_matrix = self._get_jump_matrix(self)

        self.learner = SequenceLearner(
            self.model,

        )

    def _get_jump_matrix(self):
        sigma = str(sigma_lvl).replace(".", "_")
        dir_name = f"sigma_{sigma}"
        jump_path = os.path.join(self.data_path, dir_name, "jump_matrix.csv")
        return np.loadtxt(
            jump_path, delimiter=","
        )

    def _get_chains(self):
        sigma = str(sigma_lvl).replace(".", "_")
        dir_name = f"sigma_{sigma}"
        chains_path = os.path.join(self.data_path, dir_name, "chains")
        
        chains = []
        for chain_name in os.listdir(chains_path):
            chain_path = os.path.join(chains_path, chain_name)
            chain_df = pd.read_csv(chain_path)
            chains.append(chain_df)
        return chains

    

    # def plot_neurons(self):
    #     sns.set_style("white")
    #     plt.figure(figsize=(6, 4))
    #     legend_handles = self.wiring.draw_graph(
    #         draw_labels=True, neuron_colors={"command": "tab:cyan"}
    #     )
    #     plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    #     sns.despine(left=True, bottom=True)
    #     plt.tight_layout()
    #     plt.show()

class NCPModel:
    def __init__(
        self,
        neuron_units: int = 16,
        out_features: int = 1,
        lr: float = 0.005,
        batch_size: int = 1,
        validation_percent: float = 0.15,
        data_path: str = "data/synthetic_data/sigma_0_5/chains/chain_0.csv"
    ):
        self.wiring = AutoNCP(neuron_units, out_features)
        self.ltc_model = CfC(1, self.wiring)
        self.learn = SequenceLearner(self.ltc_model, lr=lr, data_path=data_path, batch_size=batch_size, validation_percent=validation_percent)
        self.trainer = pl.Trainer(
            logger=[
                pl.loggers.CSVLogger("log", name="disease_ncp"),
                pl.loggers.TensorBoardLogger("log", name="disease_ncp")
            ],
            max_epochs=400,
            gradient_clip_val=1,  # Clip gradient to stabilize training
            accelerator="auto",
            devices="auto"
        )
        self.data_path = data_path

    def plot_neurons(self):
        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(
            draw_labels=True, neuron_colors={"command": "tab:cyan"}
        )
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    # def visualize(self, first_n: int = 100):
    #     sns.set()
    #     data_x = self.learn.data_x.numpy().flatten()
    #     data_t = self.learn.data_t.numpy().flatten()

    #     cumsum_t = np.cumsum(data_t)
    #     last_time = cumsum_t[-1]

    #     time_data = np.linspace(0, last_time, 1000).reshape([1, -1, 1]).astype(np.float32)
    #     tensor_time_data = torch.tensor(time_data)

    #     with torch.no_grad():
    #         prediction = self.ltc_model(tensor_time_data)[0].numpy()
    #     plt.figure(figsize=(6, 4))

    #     plt.plot(data_t[0, :first_n, 0], data_x[0, :first_n, 0], label="Target output")

    #     plt.plot(time_data[0, :, 0], prediction[0, :, 0], label="NCP output")
    #     plt.title("Before training")
    #     plt.xlabel("Time")
    #     plt.ylabel("Emission")
    #     plt.legend(loc="upper right")
    #     plt.show()

    def train(self):
        self.trainer.fit(self.learn, self.learn.train_dataloader, self.learn.val_dataloader)

    def test(self):
        self.trainer.test(self.learn, self.learn.val_dataloader)


if __name__ == "__main__":
    ncp = NCPModel()
    # ncp.plot_neurons()
    ncp.learn.visualize_data()
    # ncp.visualize()
    # ncp.train()
