# TODO

1. configs

- chia thành 2 dirs
    - aa_configs
    - smiles_configs
- tạo một file `build_config.py` để build config (**Cân nhắc**)

2. data

- chia thành 2 dirs
    - aa_data: chứa data liên quan đến amino acid
    - sp_data: chứa data liên quan đến signal protein
        - train_set.fasta
        - uniprot_sprot.fasta
        - benchmark_set_sp5.fasta
- `data_utils.py`
- `sp_dataset.py`

3. lightning_module

- sửa lại theo template:
    - data_module: prepare_data, split, loader
    - module: define model, metrics, loss func,...; training/validation/testing/predict(_nếu cần thiết_) progess và một
      số hàm phụ dùng để lưu results

4. model

- `nn_layers.py`
- `sp_<model>.py`
- tạo thêm một file `model_utils.py`: load_model, freeze_layer, unfreeze_layer, load_config (_nếu cần thiết_)

5. callbacks

- đưa file `callback_utils.py` vào thư mục callbacks

6. out

- gồm 3 dirs
    - đầu ra dự đoán của model: `results`
    - kết quả các độ đo: `metrics`
    - hình ảnh visualize `figure`: data, metrics, wrong predict label, training/validation loss

7. `visualization.py`

- visualize data: (**có thể sử dụng pycharm notebook**)
- visualize kết quả độ đo trên từng loài, từng nhãn
- visualize kết quả sai trên từng loài, từng nhãn

8. checkpoints

- transformer: base - medium - heavy
- cnn: base - medium - heavy
- lstm/bi-lstm: base - medium - heavy
- bert: protbert - pretrained protbert - pretrained protbert with freezing embedding layer
- **Note: Đánh giá xem bao nhiêu tham số (dung lượng bao nhiêu) để được đánh giá là medium or heavy**