from typing import TypedDict

class EncoderNCPConfig(TypedDict):
    n_units:int

class DecoderNCPConfig(TypedDict):
    n_units:int

class EncoderDecoderConfig(TypedDict):
    encoder_config:EncoderNCPConfig
    decoder_config:DecoderNCPConfig
    latent_dim:int

class NCPTrainingConfig(TypedDict):
    model_config:EncoderDecoderConfig
    learning_rate:float
    validation_chains:int
    data_path:str
    sigma_lvl:float
    batch_size:int


DEFAULT_NCP_TRAINING_CONFIG = NCPTrainingConfig(
    model_config = EncoderDecoderConfig(
        encoder_config = EncoderNCPConfig(
            n_units = 16
        ),
        decoder_config = DecoderNCPConfig(
            n_units = 16
        ),
        latent_dim = 128
    ),
    learning_rate = .005,
    validation_chains = 1,
    data_path = "data/synthetic_data",
    sigma_lvl = 0.5,
    batch_size = 1,
)