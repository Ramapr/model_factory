from pydantic import BaseModel, field_validator
from fastapi import UploadFile
from pydantic import ConfigDict



class CAutoencoderParam(BaseModel):
    lr: float = 1e-3
    hidden_dim: int | None = None
    botlneck: int | None = None
    skip_ouput_features: int


class AutoencoderParam(BaseModel):
    lr: float = 1e-3
    hidden_dim: int | None = None
    botlneck: int | None = None

    @field_validator('hidden_dim')
    @classmethod
    def validate_hm(cls, hidden_dim: int ) -> int:
        if hidden_dim < 2:
            raise ValueError('Hidden dim cannot be  less than 2')
        return hidden_dim

    @field_validator('botlneck')
    @classmethod
    def validate_btn(cls, botlneck: int ) -> int:
        if botlneck < 2:
            raise ValueError('botlneck dim cannot be  less than 2')
        return botlneck


class NormScaleParam(BaseModel):
    scale: bool = True
    norm: bool = False


class DataPrepareParams(BaseModel):
    use_filter_in_table: bool = False
    train_test_split: float = 0.8
    type_of_split: str = 'random'
    bs: int = 64
    workers: int = 0
    preprocessing: NormScaleParam

    @field_validator('type_of_split')
    @classmethod
    def validate_mode(cls, type_of_split: str ) -> str:
        if type_of_split not in ['random', 'last']:
            raise ValueError()
        return type_of_split


class ModelParams(BaseModel):
    params : AutoencoderParam | CAutoencoderParam


class EarlyStop(BaseModel):
    min_delta: float = 0.01
    patience: int = 3
    mode: str = "min"

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, mode: str ) -> str:
        if mode not in ['min', 'max']:
            raise ValueError()
        return mode

class CheckPoint(BaseModel):
    save_top_k : int = 5
    save_weights_only : bool = False
    # change here to True because we save at the end best model (think about it)


class Trainer(BaseModel):
    log_every: int = 2
    max_epochs: int = 4
    enable_checkpointing: bool = True # move out


class Mlflow(BaseModel):
    exp_name: str
    run_name: str

    @field_validator('exp_name')
    @classmethod
    def validate_experiment_name(cls, exp_name: str) -> str:
        return exp_name

    @field_validator('run_name')
    @classmethod
    def validate_run_name(cls, run_name: str) -> str:
        return run_name


class InferenceData(BaseModel):
    same_as_train: True
    preprocessing: NormScaleParam | None


class ConfigRun(BaseModel):
    dataprepare: DataPrepareParams
    model: ModelParams
    early_stop: EarlyStop
    chkp: CheckPoint
    trainer: Trainer
    mlflow: Mlflow
    inf_data: InferenceData | None = None

class DataURL(BaseModel):
    pass

class TrainRequest(BaseModel):
    username: str
    config: ConfigRun
    data: DataURL
    test_data: DataURL


class ModelArtifactsURL(BaseModel):
    pass
    # add validation


class ModelURI(BaseModel):
    pass
    # add validation


class ExportModel(BaseModel):
    artifacts_uri: str
    export_type: str = 'best'

    @field_validator('artifacts_uri')
    @classmethod
    def validate_artifacts_uri(cls, artifacts_uri: str) -> str:
        # f's3://mlflow-artifacts/{proj_num}/{run_id}/artifacts'
        host, path = artifacts_uri.split('//')
        if not host.startswith('s3'):
            raise ValueError()
        if not path.startswith('mlflow-artifacts'):
            raise ValueError()
        if not path.endswith('artifacts'):
            raise ValueError()
        if path.split('/') != 3:
            raise ValueError()
        return artifacts_uri


    @field_validator('export_type')
    @classmethod
    def validate_export_type(cls, export_type: str) -> str:
        if export_type not in ['best', 'last']:
            raise ValueError('text')
        return export_type
