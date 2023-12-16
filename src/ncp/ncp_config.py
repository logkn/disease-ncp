from typing import Optional, TypedDict


class EncoderNCPConfig(TypedDict):
    n_units: int


class DecoderNCPConfig(TypedDict):
    n_units: int


class EncoderDecoderConfig(TypedDict):
    encoder_n_units: int
    decoder_n_units: int
    latent_dim: int


class DataConfig(TypedDict):
    data_path: str
    validation_pct: float
    batch_size: int
    train_samples: int
    val_samples: int


class NCPTrainingConfig(TypedDict):
    model_config: EncoderDecoderConfig
    data_config: DataConfig
    learning_rate: float
    sigma_lvl: Optional[float]


DEFAULT_NCP_TRAINING_CONFIG = NCPTrainingConfig(
    model_config=EncoderDecoderConfig(
        encoder_n_units=272, decoder_n_units=272, latent_dim=256
    ),
    data_config=DataConfig(
        data_path="data/synthetic_data",
        validation_pct=0.2,
        batch_size=512,
        train_samples=50000,
        val_samples=5000,
    ),
    learning_rate=0.008,
    # sigma_lvl=0.5,
)
