import os
import torch.cuda
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar, Callback
from pytorch_lightning.loggers import WandbLogger

from models.Pix2Pix import Pix2Pix
from models.PlateDataModule import PlateDataModule

wandb_logger = WandbLogger(project="pix2pix")


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            n = 8
            a, b = batch

            wandb_logger.log_image(key='source', images=[img for img in a[:n]])
            wandb_logger.log_image(key='generated', images=[img for img in outputs[:n]])
            wandb_logger.log_image(key='origin', images=[img for img in b[:n]])


if __name__ == '__main__':
    DATA_PATH = 'data/plates2/train'
    BATCH_SIZE = 32 + 16
    NUM_WORKERS = int(os.cpu_count() / 2)
    PATCH = (1, 256 // 2 ** 4, 256 // 2 ** 4)
    SAVING_DIR = 'saving_ckpt'

    # Callbacks
    chk_callback = ModelCheckpoint(
        dirpath=SAVING_DIR,
        filename='pix2pix_{epoch:02d}-{lpips:.3f}',
        verbose=True,
        save_top_k=5,
        monitor='lpips',
        mode='min'
    )
    early_stop_callback = EarlyStopping(monitor='lpips', min_delta=0.00, patience=15, verbose=True, mode='min')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    log_predictions_callback = LogPredictionsCallback()

    dm = PlateDataModule(DATA_PATH, BATCH_SIZE, NUM_WORKERS)
    model = Pix2Pix(batch_size=BATCH_SIZE, patch=PATCH)
    trainer = Trainer(
        accelerator="auto",
        callbacks=[chk_callback, early_stop_callback, lr_monitor_callback, log_predictions_callback, RichProgressBar()],
        devices=1 if torch.cuda.is_available() else None,
        precision=16,
        amp_backend='apex',
        max_epochs=300,
        logger=wandb_logger
    )
    trainer.tune(model)
    trainer.fit(model, dm)
