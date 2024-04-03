import sys
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import params
import utils as ut


class CustomTQDMProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


tqdm_progress_bar = CustomTQDMProgressBar()

rich_progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    )
)

model_checkpoint = ModelCheckpoint(
    dirpath=str(Path(ut.ROOT_DIR) / 'checkpoints'),
    filename=f"{params.MODEL}_{params.DATA}_epoch={params.EPOCHS}_{params.ENV}",
    enable_version_counter=True,
    monitor='val_loss',
    every_n_epochs=1,
    save_on_train_epoch_end=True,
    save_top_k=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="max"
)
