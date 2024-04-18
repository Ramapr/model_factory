# server imports
import json
import os
from io_utils import save_data
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile

from dataclass import ConfigRun, DataURL, ExportModel

from utils import check_env_var, drop_files_from_dir, get_list_of_configs, select_model
from train_utils import ExpTracker
from export_utils import load_model, load_model_artifacts, dump_model


app = FastAPI(
    title="ML models factory 🤖 ",
    description="Сервис для построения и обучения мл моделей",
    version="0.0.1",
)


@app.on_event("startup")
async def startup_event():
    """
    check:
    1.  variables
    2.  path  roots
    3.  mlflow uri
    """
    pass
    # check_env_var()
    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[0]
    # create_dir(ROOT, "tmp"))
    # create_dir(ROOT, "runs"))
    # cached_path = create_dir(ROOT, "cache"))
    # os.environ['export_cache'] = cached_path
    # create_dir(ROOT, "tmp"))


@app.get("/all_models_configs")
async def get_all_models_configs():
    return {"message": "ok", "content": get_list_of_configs()}


@app.get("/config/{model_name}")
async def get_models_configs(model_name: str):
    configs = get_list_of_configs()
    if model_name not in configs.keys():
        raise HTTPException(status_code=404, detail=f"config for '{model_name}' not found")
    return {"message": "ok", "content": configs[model_name]}


def train_func(*args):
    # remove
    pass
    # model = select_model()(  config.params )
    # main_train_function()


@app.post("/train/link_files/{model_name}")
async def train(model_name:str,
                username: str,
                cfg: ConfigRun,
                data_link: str,  # DataURL,
                test_link: str,  # DataURL,
                background_tasks: BackgroundTasks,
):
    path = os.environ["tmp_dir"]
    train_path = save_data(data_link, path)
    if test_link:
        test_path = save_data(test_link, path)

    exp_track = ExpTracker(
        exp_name=cfg.mlflow.exp_name, run_name=cfg.mlflow.run_name, username=username
    )
    experiment_link = exp_track.get_link()
    data_paths = {"train_path": train_path, "test_path": test_path}
    background_tasks.add_task(train_func, exp_track, cfg, data_paths)
    return {"message": "ok", "url": experiment_link}


@app.post("/train/upload_files/{model_name}")
async def train_on_file(model_name:str,
                        username: str,
                        cfg: ConfigRun,
                        data_file: UploadFile,
                        background_tasks: BackgroundTasks,
                        test_file: Optional[UploadFile] = None,
):
    path = os.environ["tmp_dir"]
    train_path = save_data(data_file, path)
    if test_file:
        test_path = save_data(test_file, path)

    exp_track = ExpTracker(
        exp_name=cfg.mlflow.exp_name, run_name=cfg.mlflow.run_name, username=username
    )
    experiment_link = exp_track.get_link()

    data_paths = {"train_path": train_path, "test_path": test_path}
    background_tasks.add_task(train_func, exp_track, cfg, data_paths)
    return {"message": "ok", "url": experiment_link}


@app.post("/run")
async def run(model_link: str, val_data_link: str):
    print(model_link)
    print(val_data_link)
    # local_artifact_path = load_model_artifacts(params)
    # model = load_model(local_artifact_path)

    pass


@app.post("/export/{project_id}/{model_hash}")
async def export(params: ExportModel):
    try:
        local_artifact_path = load_model_artifacts(params)
        model = load_model(local_artifact_path)
        model_in_bytes = dump_model(model, local_artifact_path)
        drop_files_from_dir(os.environ["cache_dir"])
        return {'message': 'ok', 'file_bytes': str(model_in_bytes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'error {e}')
