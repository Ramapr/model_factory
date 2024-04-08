import mlflow
import torch

from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.mlflow import MLFlowLogger



def get_loaders(train_set,
                train_test_split: float,
                splittype: str,
                bs: int,
                workers: int
                ):
    # split the train set into two
    train_set_size = int(len(train_set) * \
                         train_test_split)
    valid_set_size = len(train_set) - train_set_size

    seed = torch.Generator().manual_seed(42)

    if splittype == 'random':
        train_set, valid_set = random_split(train_set,
                                            [train_set_size, valid_set_size],
                                            generator=seed)

    train_loader = DataLoader(train_set,
                              batch_size=bs,
                              num_workers=workers)

    valid_loader = DataLoader(valid_set,
                              batch_size=bs,
                              num_workers=workers)

    x_ex, y_ex = train_set.__getitem__(0)
    in_out_signature = {'x': x_ex.numpy()[None, :],
                        'y': y_ex.numpy()[None, :]}

    return {'train':train_loader, 'val':valid_loader, 'signature':in_out_signature}


class ExpTracker:
    def __init__(self,
                 exp_name: str,
                 run_name: str,
                 username: str):

        self.ml = mlflow
        self.ml.set_tracking_uri(credits["URL"]) # os.get
        self.ml.set_experiment(exp_name)
        self.ml.pytorch.autolog()
        self.ml.start_run(run_name=run_name)
        # path_artifacts = mlflow.active_run().info.artifact_uri
        # NOTE:   for server part
        self.ml.set_tag("mlflow.user", username)
        #mlflow.active_run().info.user_id =
        self.url = self.ml.get_tracking_uri()
        self.run_id = self.ml.active_run().info.run_id
        self.exp_id = self.ml.active_run().info.experiment_id
        self.exp_name = self.ml.get_experiment(self.ml.active_run().info.experiment_id).name

    def get_link(self) -> str:
        return f"{self.url}/#/experiments/{self.exp_id}/runs/{self.run_id}"


    def get_setups(self) -> dict:
        return {"experiment_name": self.exp_name,
                "tracking_uri": self.url,
                "run_id ": self.run_id
                }

    def stop(self):
        self.ml.end_run()




def main_train_function(config,
                        model,
                        mlflow_setups,
                        train_loader,
                        valid_loader):

    checkpoints_dir

    mlf_logger = MLFlowLogger(**{**mlflow_setups, **{'log_model':"all"}})
        # experiment_name= # mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        # tracking_uri=mlflow.get_tracking_uri(),
        # run_id=mlflow.active_run().info.run_id,
        # log_model=config.mlflow.type,
        # )

    e_stop = EarlyStopping(monitor=config.early_stop.monitor ,
                           min_delta=config.early_stop.min_delta ,
                           patience=config.early_stop.patience,
                           verbose=config.early_stop.verbose,
                           mode=config.early_stop.mode)

    # TRAIN
    checkpoint_callback = ModelCheckpoint(dirpath=config.chkp.path,
                                          monitor=config.chkp.monitor,
                                          filename=config.chkp.filename_template,
                                          save_top_k=config.chkp.save_top_k,
                                          save_weights_only=config.chkp.save_weights_only
                                          )

    trainer = Trainer(auto_scale_batch_size=config.trainer.auto_scale_batch_size,
                      auto_lr_find=config.trainer.auto_lr_find,
                      devices=config.trainer.devices,
                      accelerator=config.trainer.accelerator,
                      log_every_n_steps=config.trainer.log_every,
                      max_epochs=config.trainer.max_epochs,
                      enable_checkpointing=config.trainer.enable_checkpointing,
                      default_root_dir=checkpoints_dir,
                      logger=mlf_logger,
                      callbacks=[e_stop, checkpoint_callback]
                      )

    trainer.fit(model, train_loader, valid_loader)

    mlf_logger.experiment.log_artifact(run_id=mlf_logger.run_id,
                                       local_path=checkpoint_callback.best_model_path)  # , infer_signature())

    mlf_logger.experiment.log_dict(run_id=mlf_logger.run_id,
                                   dictionary={**params_info, **prep_info, **model_layers},
                                   artifact_file='info.json')

    x_ex, y_ex = train_set.__getitem__(0)
    signature = infer_signature(x_ex.numpy()[None, :], y_ex.numpy()[None, :])
