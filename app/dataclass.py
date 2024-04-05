from pydantic import BaseModel


class CAutoencoderParam(BaseModel):
    lr: float = 1e-3
    hidden_dim: int | None = None
    botlneck: int | None = None
    skip_ouput_fetures: int


class AutoencoderParam(BaseModel):
    lr: float = 1e-3
    hidden_dim: int | None = None
    botlneck: int | None = None


class DataPrepareParams(BaseModel):
    use_filter_in_table: bool = False
    train_test_split: float = 0.8
    type_of_split: str = 'random'
    bs: int = 64
    workers: int = 0
    scale: bool = True
    norm: bool = False

    def validate_splits():
       pass


class ModelParams(BaseModel):
    params : AutoencoderParam | CAutoencoderParam


class EarlyStop(BaseModel):
    min_delta: float = 0.01
    patience: int = 3
    mode: str = "min"

    def validate_mode():
        pass

class CheckPoint(BaseModel):
  save_top_k : int = 5
  save_weights_only : bool = False
  # change here to True because we save at the end best model (think about it)


class Trainer(BaseModel):
  log_every: int = 2
  max_epochs: int = 4
  enable_checkpointing: bool = True


class Mlflow(BaseModel):
    exp_name: str
    run_name: str

    def validate_experiment_name():
        pass

    def validate_run_name():
        pass


class ConfigRun(BaseModel):
    dataprepare: DataPrepareParams
    model: ModelParams
    early_stop: EarlyStop
    chkp: CheckPoint
    trainer: Trainer
    mlflow: Mlflow


class DataURL(BaseModel):
    pass


class ModelArtifactsURL(BaseModel):
    pass
    # add validation

class ModelURI(BaseModel):
    pass
    # add validation