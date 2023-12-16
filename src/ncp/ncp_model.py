import os
import random
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import seaborn as sns
import torch
import torch.nn as nn
from ncp_config import DEFAULT_NCP_TRAINING_CONFIG, NCPTrainingConfig
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    BatchSizeFinder,
    RichProgressBar,
    LearningRateFinder,
    StochasticWeightAveraging,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from optuna.integration import PyTorchLightningPruningCallback

pl.seed_everything(42)

torch.set_float32_matmul_precision("medium")


class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # Define AutoNCP wiring for the encoder
        encoder_wiring = AutoNCP(latent_dim, int(0.3 * latent_dim))
        self.latent_dim = latent_dim
        self.rnn = CfC(2, encoder_wiring)

    def forward(self, x, dt):
        combined_input = torch.cat((x, dt), dim=-1)
        _, hidden_state = self.rnn(combined_input)
        return hidden_state


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        # Define AutoNCP wiring for the decoder
        decoder_wiring = AutoNCP(hidden_size, int(0.3 * hidden_size))
        self.rnn = CfC(1, decoder_wiring, proj_size=1)

    def forward(self, hidden_state, dt):
        decoder_output, _ = self.rnn(dt, hx=hidden_state)
        return decoder_output


class EncoderDecoder(pl.LightningModule):
    def __init__(self, config: NCPTrainingConfig, trial: optuna.trial.Trial = None):
        super().__init__()
        self.config = config
        self.tuning = config["tuning"]
        self.trial = trial

        self._init_hyperparameters()

        self._init_models()

        self._load_datasets()

        self._load_trainer()

        self.save_hyperparameters()

    def _objective(trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna to minimize.

        :param trial: Optuna trial.
        :return: Validation loss.
        """
        config = DEFAULT_NCP_TRAINING_CONFIG

        config["latent_dim"] = trial.suggest_int("latent_dim", 24, 1024)
        config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        config["data_config"]["batch_size"] = trial.suggest_int(
            "batch_size", 128, 2048 * 4
        )

        config["tuning"] = True

        enc_dec = EncoderDecoder(config, trial)
        enc_dec.training_loop()
        return enc_dec.trainer.callback_metrics["val_loss"].item()

    def start_optuna_study(n_trials: int = 100):
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            study_name="hyperparam_tuning",
        )
        study.optimize(EncoderDecoder._objective, n_trials=n_trials, timeout=60*6, show_progress_bar=True)
        return study

    def _load_trainer(self):
        """
        Initializes the PyTorch Lightning trainer.
        """
        trainer_callbacks = [
            StochasticWeightAveraging(swa_lrs=1e-3),
            RichProgressBar(),
        ]

        if self.tuning:
            trainer_callbacks.append(OptunaPruning(self.trial, monitor="val_loss"))
        else:
            trainer_callbacks.append(EarlyStopping(monitor="val_loss", patience=5))

        self.trainer = pl.Trainer(
            logger=[
                pl.loggers.TensorBoardLogger("lightning_logs/"),
                pl.loggers.CSVLogger("lightning_logs/"),
            ],
            precision="bf16-mixed",
            max_epochs=25,
            gradient_clip_val=1,
            log_every_n_steps=5,
            callbacks=trainer_callbacks,
        )

    def training_loop(self):
        self.trainer.fit(
            self,
            train_dataloaders=self.train_dataloader(),
            val_dataloaders=self.val_dataloader(),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self._load_datasets()
        return self.train_dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        self._load_datasets()
        return self.val_dl

    def _init_models(self):
        """
        Initializes the encoder and decoder models.
        """

        self.latent_dim = self.config["latent_dim"]

        self.encoder = Encoder(
            self.latent_dim,
        )  # Adjust sizes as needed
        self.decoder = Decoder(self.latent_dim, 1)  # Assuming dt is 1D

    def _init_hyperparameters(self):
        """
        Initializes the hyperparameters from the config file.
        """

        self.hparams["lr"] = self.hparams[
            "learning_rate"
        ] = self.learning_rate = self.config["learning_rate"]
        self.data_path = self.config["data_config"]["data_path"]

        self.hparams["validation_pct"] = self.validation_pct = self.config[
            "data_config"
        ]["validation_pct"]
        self.hparams["batch_size"] = self.batch_size = self.config["data_config"][
            "batch_size"
        ]

        self.train_samples = self.config["data_config"]["train_samples"]
        self.val_samples = self.config["data_config"]["val_samples"]

        self.sigma_lvl = self.config.get("sigma_lvl", None)

    def forward(self, x_encoder, dt_encoder, dt_decoder):
        hidden_state = self.encoder(x_encoder, dt_encoder)
        prediction = self.decoder(hidden_state, dt_decoder)
        return prediction

    def training_step(self, batch, batch_idx):
        x_encoder, dt_encoder, dt_decoder, y = batch
        y_pred = self(x_encoder, dt_encoder, dt_decoder)
        loss = nn.MSELoss()(y_pred, y)
        self.log("train_loss", loss)  # Logging to TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        x_encoder, dt_encoder, dt_decoder, y = batch
        y_pred = self(x_encoder, dt_encoder, dt_decoder)
        val_loss = nn.MSELoss()(y_pred, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _get_chains(self) -> list[pd.DataFrame]:
        """
        Returns a list of chains as pandas DataFrames from the given path.

        :return: List of chains as pandas DataFrames.
        """
        paths = []

        if self.sigma_lvl:
            chains_path = os.path.join(
                *os.path.split(self.data_path),
                "sigma_" + str(self.sigma_lvl).replace(".", "_"),
                "chains"
            )
            paths.append(chains_path)
        else:
            for sigma_lvl in os.listdir(self.data_path):
                chains_path = os.path.join(
                    *os.path.split(self.data_path), sigma_lvl, "chains"
                )
                paths.append(chains_path)

        chains = []
        for chains_path in paths:
            for chain_file in os.listdir(chains_path):
                if chain_file.endswith(".csv"):
                    chain = pd.read_csv(os.path.join(chains_path, chain_file))
                    chains.append(chain)
        return chains

    def _train_val_split(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Splits a list of chains into a training and validation set.

        :param dfs: List of chains as pandas DataFrames.
        :return: Tuple of training and validation sets.
        """
        dfs = self._get_chains()
        val_pct = self.validation_pct
        val_size = int(len(dfs) * val_pct)
        val_dfs = dfs[:val_size]
        train_dfs = dfs[val_size:]
        return train_dfs, val_dfs

    def _sample_from_dfs(
        self,
        dataframes: list[pd.DataFrame],
        num_samples: int,
        m_range: Tuple[int, int],
        n_range: Tuple[int, int],
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Samples data from a list of chains, and returns (encoder_x, encoder_dt, decoder_dt, y).

        Each sample is a sequence of observations, where
        - The chain is randomly selected from the list of chains.
        - The encoding sequence length is randomly selected from the range m_range.
        - The number of additional observations for decoding is randomly selected from the range n_range.
            - In total, the decoder sequence length is m + n.
        - The encoder and decoder sequences are contiguous from the same chain.

        Example: Suppose m=3 and n=2, and we start at observation 5 in the chain.
        Then:
            - encoder_x = [x_5, x_6, x_7]
            - encoder_dt = [dt_5, dt_6, dt_7]
            - decoder_dt = [dt_5, dt_6, dt_7, dt_8, dt_9]
            - y = [x_5, x_6, x_7, x_8, x_9]
        """

        encoder_x, encoder_dt, decoder_dt, y = [], [], [], []

        with tqdm(total=num_samples, desc="Sampling from chains") as pbar:
            while len(encoder_x) < num_samples:
                df = random.choice(dataframes)
                length = len(df)

                m = np.random.randint(*m_range)
                n = np.random.randint(*n_range)
                m_n = min(m + n, length)

                try:  # avoid negative start_idx by retrying if ValueError
                    start_idx = np.random.randint(0, max(0, length - m_n))
                except ValueError:
                    continue

                encoder_x.append(
                    torch.tensor(
                        df["emission"][start_idx : start_idx + m].values,
                        dtype=torch.float,
                    )[:, None]
                )
                encoder_dt.append(
                    torch.tensor(
                        df["time"][start_idx : start_idx + m].values, dtype=torch.float
                    )[:, None]
                )
                decoder_dt.append(
                    torch.tensor(
                        df["time"][start_idx : start_idx + m_n].values,
                        dtype=torch.float,
                    )[:, None]
                )
                y.append(
                    torch.tensor(
                        df["emission"][start_idx : start_idx + m_n].values,
                        dtype=torch.float,
                    )[:, None]
                )
                pbar.update(1)
        return encoder_x, encoder_dt, decoder_dt, y

    def _load_datasets(
        self, m_range: Tuple[int, int] = (50, 400), n_range: Tuple[int, int] = (10, 200)
    ):
        """
        Loads the training and validation dataloaders

        :param m_range: Range of encoder sequence lengths.
        :param n_range: Range of decoder sequence lengths (not including the encoder sequence length)
        """
        if hasattr(self, "train_dl") and hasattr(self, "val_dl"):
            return

        train_dfs, val_dfs = self._train_val_split()

        train_tuple = self._sample_from_dfs(
            train_dfs, self.train_samples, m_range, n_range
        )
        val_tuple = self._sample_from_dfs(val_dfs, self.val_samples, m_range, n_range)

        # Pad sequences
        train_tuple = tuple(pad_sequence(seq, batch_first=True) for seq in train_tuple)

        val_tuple = tuple(pad_sequence(seq, batch_first=True) for seq in val_tuple)

        train_dataset, val_dataset = TensorDataset(*train_tuple), TensorDataset(
            *val_tuple
        )

        train_dataloader, val_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        ), DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
        )

        self.train_dl, self.val_dl = train_dataloader, val_dataloader

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


if __name__ == "__main__":
    study = EncoderDecoder.start_optuna_study(n_trials=50)

    # Save study
    study_name = "hyperparam_tuning"
    study_path = os.path.join("data", study_name)
    study.trials_dataframe().to_csv(os.path.join(study_path, "trials.csv"))
