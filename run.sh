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

# testing
python3 test.py \
        --checkpoint "cnn_aa_epoch=100_default_kaggle.ckpt" \
        --batch_size 8