from typing import TypedDict

class EncoderNCPConfig(TypedDict):
    n_units:int

class DecoderNCPConfig(TypedDict):
    n_units:int

class EncoderDecoderConfig(TypedDict):
    encoder_n_units:int
    decoder_n_units:int
    latent_dim:int

class NCPTrainingConfig(TypedDict):
    model_config:EncoderDecoderConfig
    learning_rate:float
    n_validation_chains:int
    data_path:str
    sigma_lvl:float
    batch_size:int


DEFAULT_NCP_TRAINING_CONFIG = NCPTrainingConfig(
    model_config = EncoderDecoderConfig(
        encoder_n_units = 64,
        decoder_n_units = 64,
        latent_dim = 50
    ),
    learning_rate = .005,
    n_validation_chains = 1,
    data_path = "data/synthetic_data",
    sigma_lvl = 0.5,
    batch_size = 1,
)