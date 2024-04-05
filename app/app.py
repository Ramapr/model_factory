# server imports
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import BackgroundTasks
#from starlette.datastructures import UploadFile as upf
import json
from time import sleep
import os

from dataclass import ConfigRun, DataURL, ExportModel
from utils import get_list_of_configs
from utils import check_env_var
from utils import drop_files_from_dir
from mlflow_utils import load_model_artifacts, get_params, select_model, load_checkpoint, dump_model

from pathlib import Path

app = FastAPI(title="ML models factory 🤖 ",
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
    #check_env_var()
    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[0]
    # create_dir(ROOT, "tmp"))
    # create_dir(ROOT, "runs"))
    # cached_path = create_dir(ROOT, "cache"))
    # os.envsetup['export_cache'] = cached_path


@app.get("/models_configs")
async def get_models_configs():
    return {'message': 'ok', 'content': json.dump(get_list_of_configs())}


def train_func(**kv):
    sleep(15)
    print(kv)
    sleep(15)
    pass


@app.post("/train")
async def train(username: str,
                cfg: ConfigRun,
                data_link: str, #DataURL,
                test_link: str, #DataURL,
                background_tasks: BackgroundTasks):

    #pass
    #cfg
    kv = {'test': 0.1, 'train': 10}
    background_tasks.add_task(train_func, kv=kv)


@app.post("/train/file/")
async def train(username: str,
                cfg: ConfigRun,
                data_file: UploadFile,
                test_file: UploadFile,
                background_tasks: BackgroundTasks):

    #pass
    #cfg
    kv = {'test': 0.1, 'train': 10}
    background_tasks.add_task(train_func, kv=kv)


@app.post("/run")
async def run(model_link: str,
              val_data_link: str
              ):
    pass

@app.post("/export")
async def export(params: ExportModel):
    pass
    try:
        local_artifact_path = load_model_artifacts(params)
        model_dict = get_params(local_artifact_path)
        model = select_model(model_dict['model_type'])(**model_dict['layers'])

        checkpoint = load_checkpoint(local_artifact_path)
        model.load_state_dict(checkpoint["state_dict"])
        model_in_bytes = dump_model(model, local_artifact_path)

        drop_files_from_dir(os.environ["cache_dir"])
        return {'message': 'ok', 'file_bytes': str(model_in_bytes) }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'error {e}')
