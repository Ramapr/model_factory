# server imports
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile

from starlette.datastructures import UploadFile as upf
import json

from utils import get_list_of_configs
from utils import check_env_var

app = FastAPI(title="ML models factory 🤖 🏭 ",
    description="Сервис для построения и обучения мл моделей",
    version="0.0.1",
)


@app.on_event("startup")
async def startup_event():
    """
    """
    check_env_var()


@app.get("/models_configs")
async def get_models_configs():
    return {'message': 'ok', 'content': json.dump(get_list_of_configs())}


@app.post("/export/{proj_num}/{run_id}")
async def export(proj_num: str,
                 run_id: str,
                 exptype: str = 'best'):
    pass


@app.post("/train/")
async def ctrain(user_id: str,
                 config_file: UploadFile,
            #    data_link: str or None = None,
            #    test_link: str or None = None,
                 data_file: UploadFile  = None,
                 test_file: UploadFile  = None,
                ):
    pass


@app.post("/run/")
async def run(model_link: str,
              val_data_link: str
              ):
    pass