# =================================================================================================
# --num_workers on training process need to set to more than 1 only if train not on personal laptop
#
# ==================================================================================================

# Training and Validation
python3 train.py \
        --model_type "cnn" \
        --data_type "aa" \
        --conf_type "default" \
        --epochs 10 \
        --batch_size 8 \
        --num_workers 2 \
        --lr 1e-6

# TODO: Pending...
python main.py fit \
       --trainer.accelerator "auto" \
       --trainer.devices "auto" --trainer.logger lightning.pytorch.loggers.WandbLogger \
       --trainer.logger.save_dir "logs" \
       --trainer.logger.name "wandb" \
       --trainer.logger.project "SPPredictor" \
       --trainer.val_check_interval 1.0 \
       --trainer.max_epochs 10 \
       --trainer.callback []


# testing
python3 test.py \
        --checkpoint "cnn_aa_epoch=100_default_kaggle.ckpt" \
        --batch_size 8

python test.py --env "kaggle" --model_type "transformer" --data_type "aa" --conf_type "default" --epochs 100 --batch_size 8