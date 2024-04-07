import os
import yaml
import json
from dotenv import dotenv_values
import shutil
import importlib
import inspect


def get_list_of_configs(path: str = './models') -> dict:
    output = {}
    models = list(filter(lambda x: not x.startswith('__') and '.' not in x,  os.listdir(path)))

    for m in models:
        cfg_path = os.path.join(os.path.join(path, m), 'config.yaml')
        with open(cfg_path, 'r') as cfg_file:
            output[m] = yaml.safe_load(cfg_file)

    return output


def check_env_var(path: str = './models') -> bool :
    if ".env" in os.listdir(path):
        credits = dotenv_values(".env")
        # load credentials from .env file
        os.environ["AWS_ACCESS_KEY_ID"] = credits["S3_USER"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = credits["S3_PASS"]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = credits["S3"]
    else:
        try:
            os.environ["AWS_ACCESS_KEY_ID"]
            os.environ["AWS_SECRET_ACCESS_KEY"]
            os.environ["MLFLOW_S3_ENDPOINT_URL"]
        except Exception as e:
            raise KeyError(e)


def create_dir(root_path: str, dirname: str) -> str:
    dir_path = os.path.join(root_path, dirname)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'INFO : dir \'{dirname}\' created')
    return dir_path


def drop_files_from_dir(folder):
    # check here
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def write_file2fs(path, data):
    """
    open cached file to write to file system
    move to io
    """
    data_path = os.path.join(path, data.filename)
    contents = data.file.read()
    with open(data_path, 'wb') as f:
        f.write(contents)
    data.file.close()
    return data_path




