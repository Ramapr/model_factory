import os

import boto3
from fastapi import HTTPException
from starlette.datastructures import UploadFile


def load_from_s3(data: str, base_path: str, credits):
    """ """
    # credits
    # data  = "{bucket}/path/file.csv"
    BUCKET_NAME = data.split("/")[0]
    src_file_name = data.split("/")[-1]

    file_path = data.split(BUCKET_NAME)[1][1:]
    target_path = os.path.join(base_path, src_file_name)

    # fix creds to os.environ[""]
    try:
        # connect to minio
        s3 = boto3.resource(
            "s3",
            endpoint_url=credits["S3"],
            aws_access_key_id=credits["S3_USER"],
            aws_secret_access_key=credits["S3_PASS"],
        )
        s3.Bucket(BUCKET_NAME).download_file(file_path, target_path)
        return target_path

    except Exception as e:
        return None, e


def write_file2fs(path, data):
    """
    open cached file to write to file system
    """
    data_path = os.path.join(path, data.filename)
    contents = data.file.read()
    with open(data_path, "wb") as f:
        f.write(contents)
    data.file.close()
    return data_path


def save_data(data, base_path):
    if isinstance(data, str):
        return load_from_s3(data, base_path)
        # load from local s3
    else:
        # binary file
        if data.filename.split(".")[1] not in ["csv", "parquet"]:
            raise HTTPException(status_code=400, detail="unknown format for data file")

        return write_file2fs(base_path, data)
