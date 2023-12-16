from typing import Optional, TypedDict


class EncoderNCPConfig(TypedDict):
    n_units: int


class DecoderNCPConfig(TypedDict):
    n_units: int


class DataConfig(TypedDict):
    data_path: str
    validation_pct: float
    batch_size: int
    train_samples: int
    val_samples: int


class NCPTrainingConfig(TypedDict):
    latent_dim: int
    learning_rate: float
    sigma_lvl: Optional[float]
    tuning: bool


DEFAULT_NCP_TRAINING_CONFIG = NCPTrainingConfig(
    latent_dim=2,
    data_config=DataConfig(
        data_path="data/synthetic_data",
        validation_pct=0.3,
        batch_size=2048*4,
        train_samples=50000,
        val_samples=5000,
    ),
    learning_rate=0.005,
    tuning=True,
    # sigma_lvl=0.5,
)
