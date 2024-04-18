# model_factory
service

Features \ Hints :
1) use BackgroundTasks for `train/` method
2) adding new model to server = add directory to `app/models/`
3) enable `metrics/` based on Prometheus
4) enable logging based on Sentry


## methods 



# How to start

## before start

`.env`

```bash
URL="192..."
S3_USER=""
S3_PASS=""
S3="192..."
```

## localy with conda

install

```bash
$ conda create --name {name} python==3.8.5 -y
$ conda activate {name}
$ pip install -r requirements.txt
```

run app
```bash
$ cd model_factory/app
$ uvicorn app:app --reload
```

## with Docker

```bash
$ cd model_factory/app
$ docker build --tag model_factory:latest .
$ docker run ...
```

# Adding new Model

1. directory with new model name `models/` -> `models/new-model`
2. file with `Model` and `Dset` classes . **NOTE**: class of model must contain key-word `Model` and Dataset -- `Dset`.
3. add config file -- `config.yaml`


# Capabilities and limitations

