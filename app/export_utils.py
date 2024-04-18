# for export
from typing import Optional
import json
import os

import torch
from mlflow.artifacts import download_artifacts
from utils import select_model

def load_model_artifacts(params) -> str:
    return download_artifacts(
        artifact_uri=params.artifact_uri, dst_path=os.environ["cache_dir"]
    )


def get_params(path: str) -> dict:
    with open(os.path.join(path, "info.json"), mode="r") as info_f:
        info = json.load(info_f)
    return info


def load_checkpoint(path: str,  type: Optional[str] = None) -> str:
    cfiles = [i for i in os.listdir(path) if i.endswith(".ckpt")]

    if not len(cfiles):
        raise Exception("ckpt not find")
    if len(cfiles) > 1 and type is None:
        return torch.load(os.path.join(path, cfiles[0]))
    else:
        pass


def dump_model(model, path):
    model_path = os.path.join(path, "model.onnx")
    model.to_onnx(
        model_path,
        export_params=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable lenght axes
            "output": {0: "batch_size"},
        },
    )
    with open(model_path, "rb") as f:
        model_file = f.read()

    return model_file

def load_model(path):
    model_dict = get_params(path)
    model = select_model(model_dict['model_type'])(**model_dict['model_params'])
    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model