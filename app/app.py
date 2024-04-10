# server imports
# from starlette.datastructures import UploadFile as upf
import json
import os
from io import save_data
from pathlib import Path
from time import sleep

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile

from dataclass import ConfigRun, DataURL, ExportModel
from train_utils import ExpTracker
from utils import check_env_var, drop_files_from_dir, get_list_of_configs, select_model

# from export_utils import load_model_artifacts, get_params, select_model, load_checkpoint, dump_model


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


@app.get("/models_configs")
async def get_models_configs():
    return {"message": "ok", "content": json.dump(get_list_of_configs())}


def train_func(*args):
    # remove
    pass
    # model = select_model()(  config.params )
    # main_train_function()


@app.post("/train")
async def train(
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


@app.post("/train/file/")
async def train(
    username: str,
    cfg: ConfigRun,
    data_file: UploadFile,
    test_file: UploadFile | None,
    background_tasks: BackgroundTasks,
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
    pass


@app.post("/export")
async def export(params: ExportModel):
    pass

    # try:
    #     local_artifact_path = load_model_artifacts(params)
    #     model_dict = get_params(local_artifact_path)
    #     model = select_model(model_dict['model_type'])(**model_dict['layers'])

    #     checkpoint = load_checkpoint(local_artifact_path)
    #     model.load_state_dict(checkpoint["state_dict"])
    #     model_in_bytes = dump_model(model, local_artifact_path)

    #     drop_files_from_dir(os.environ["cache_dir"])
    #     return {'message': 'ok', 'file_bytes': str(model_in_bytes) }

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f'error {e}')
