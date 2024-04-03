import json
import os.path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import optim, Tensor
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import Optimizer
from torchmetrics import F1Score, MatthewsCorrCoef, AveragePrecision
from transformers import BertConfig, BertModel

import params
import utils as ut
from model.sp_bilstm import StackedBiLSTMClassifier
from model.sp_cnn import ConvolutionalClassifier
from model.sp_cnn_transformer import CNNTransformerClassifier
from model.sp_lstm import LSTMClassifier
from model.sp_transformer import TransformerClassifier


# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
# from torcheval.metrics.functional import multiclass_auroc, multiclass_auprc, multiclass_f1_score


class SPModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO: reduce weights by decreasing rate of class NO_SP (Do with last networks but results still low)
        loss_weight = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float)
        self.loss_fn = CrossEntropyLoss(weight=loss_weight)
        self.save_hyperparameters()
        # self.fabric = Fabric()

        # Load _config (Remove if unnecessary)
        # self.config = self.__load_model_config(model_type=model_type, config_path=config_path)

        # Tokenizer
        self.tokenizer = None

        # Load model
        self.model_type = params.MODEL
        self.model = self.__load_model()

        # Load metrics
        self.f1 = F1Score(task='multiclass', num_classes=len(params.SP_LABELS), average='micro')
        self.mcc = MatthewsCorrCoef(task='multiclass', num_classes=len(params.SP_LABELS))
        self.average_precision = AveragePrecision(task='multiclass', num_classes=len(params.SP_LABELS))

        # Outputs from training process
        self.validation_step_outputs_lb = []
        self.validation_step_outputs_pred = []
        self.best_val_loss = 1e6

        self.test_outputs_lb = []
        self.test_outputs_pred = []
        self.test_step_outputs = {}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.1)
        return optimizer

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any):
        loss.backward()

    def tokenize_input(self, x):
        pass

    def base_step(self, batch, batch_idx):
        x, lb, kingdom = batch
        pred = self.model(x)
        loss = self.loss_fn(pred.float(), lb.float())
        return x, lb, pred, loss, kingdom

    def training_step(self, batch, batch_idx):
        _, _, pred, loss, _ = self.base_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=params.BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        _, lb, pred, loss, _ = self.base_step(batch, batch_idx)
        self.validation_step_outputs_pred.append(pred)
        self.validation_step_outputs_lb.append(lb)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=params.BATCH_SIZE)
        # return loss

    def on_validation_epoch_end(self):
        all_pred = torch.concat(self.validation_step_outputs_pred, dim=0)
        all_lb = torch.concat(self.validation_step_outputs_lb, dim=0)

        val_loss = self.loss_fn(all_pred.float(), all_lb.float())
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # self.save_to_txt(all_pred)

        all_lb = torch.argmax(all_lb, dim=1)

        self.f1.update(all_pred, all_lb)
        self.mcc.update(all_pred, all_lb)
        self.average_precision.update(all_pred, all_lb)

        print(
            f"\nMetrics on validation set: "
            f"best_val_loss: {self.best_val_loss}, "
            f"f1: {self.f1.compute()}, "
            f"mcc: {self.mcc.compute()}, "
            f"average_precision: {self.average_precision.compute()} \n"
        )

        self.validation_step_outputs_lb.clear()
        self.validation_step_outputs_pred.clear()
        self.f1.reset()
        self.mcc.reset()
        self.average_precision.reset()

    def test_step(self, batch, batch_idx):
        _, lb, pred, loss, kingdom = self.base_step(batch, batch_idx)

        # Update outputs for calculate metrics on each class
        _pred = torch.argmax(pred, dim=1)
        _lb = torch.argmax(lb, dim=1)
        self.test_outputs_pred.extend(_pred.detach().cpu().numpy())
        self.test_outputs_lb.extend(_lb.detach().cpu().numpy())

        # Update outputs for calculate metrics on each kingdom
        for i, k in enumerate(kingdom):
            if k not in self.test_step_outputs.keys():
                self.test_step_outputs[k] = {}
                self.test_step_outputs[k]["pred"] = []
                self.test_step_outputs[k]["lb"] = []

            outputs = self.test_step_outputs[k]
            self.test_step_outputs[k]['pred'] = torch.stack([*outputs["pred"], pred[i]])
            self.test_step_outputs[k]['lb'] = torch.stack([*outputs["lb"], lb[i]])
            # outputs["pred"].append(pred[i])
            # outputs["lb"].append(lb[i])

        # self.test_step_outputs_lb.append(lb)
        # self.test_step_outputs_pred.append(pred)
        # lb = torch.argmax(lb, dim=1)

    def on_test_end(self) -> None:
        # TODO: statistic on each class (Do with each class of all outputs, in case of each class in each kingdom
        #  requires re-training (not sure, may be just adjust some lines in test_step func)

        # Apply argmax on these outputs (only for label) and evaluate the metric results
        print(classification_report(self.test_outputs_lb, self.test_outputs_pred))

        #  all_pred = torch.concat(self.test_step_outputs_lb, dim=0)
        #  all_lb = torch.concat(self.test_step_outputs_lb, dim=0)
        #  all_lb = torch.argmax(all_lb, dim=1)
        #  all_test_outputs = {}

        # Calculate metrics on each kingdom
        f1_test = []
        mcc_test = []
        average_precision_test = []
        for key, _ in params.KINGDOM.items():
            all_pred = self.test_step_outputs[key]['pred']
            all_lb = torch.argmax(self.test_step_outputs[key]['lb'], dim=1)

            # Print the statistic (the following function has ERROR about syntax)
            # self.__save_results_to_txt(all_pred.detach().cpu(), all_lb.detach().cpu(), kingdom=key)

            self.f1.update(all_pred, all_lb)
            self.mcc.update(all_pred, all_lb)
            self.average_precision.update(all_pred, all_lb)
            f1_test.append(self.f1.compute().item())
            mcc_test.append(self.mcc.compute().item())
            average_precision_test.append(self.average_precision.compute().item())

            print(
                f'\nMetrics on test set of {key}: '
                f'f1: {self.f1.compute()}, '
                f'mcc: {self.mcc.compute()}, '
                f'average_precision: {self.average_precision.compute()} \n'
            )

            self.f1.reset()
            self.mcc.reset()
            self.average_precision.reset()

        metric_dict = {
            "kingdom": params.KINGDOM.keys(),
            "F1 Score": f1_test,
            "MCC Score": mcc_test,
            "Average Precision Score": average_precision_test
        }

        self.__save_metrics_to_csv(metric_dict)

    def __load_model(self):
        """
        Load model to Lightning Module
        """
        # TODO: combine CNN + Transformer; LSTM; ProtBert pretrained:
        #  transformer and lstm models are overfit then the CNN + Trans also has low results and so do lstm
        config = self.__load_model_config(params.MODEL, params.CONF_TYPE)
        if params.MODEL == 'transformer':
            return TransformerClassifier(config)
        elif params.MODEL == 'cnn':
            return ConvolutionalClassifier(config)
        elif params.MODEL == 'st_bilstm':
            return StackedBiLSTMClassifier(config)
        elif params.MODEL == "bert_pretrained" or params.MODEL == "bert":
            return BertModel(config)
        elif params.MODEL == "lstm":
            return LSTMClassifier(config)
        elif params.MODEL == "cnn_trans":
            return CNNTransformerClassifier(config)
        else:
            return ValueError("Unknown model type")

    @staticmethod
    def __save_results_to_txt(test_prediction_results, test_true_results, kingdom):
        if not os.path.exists(ut.abspath("out")):
            os.makedirs('out', exist_ok=True)
        softmax = Softmax()
        pred_path = f'out/{kingdom}_test_prediction_results_by_{params.MODEL}.txt'
        true_path = f'out/{kingdom}_test_true_results.txt'

        # print(test_prediction_results)

        np.savetxt(ut.abspath(pred_path), softmax(test_prediction_results), fmt="%.4f")
        np.savetxt(ut.abspath(true_path), test_true_results, fmt="%d")

    @staticmethod
    def __save_metrics_to_csv(metric_dict):
        version = 0
        while os.path.exists(ut.abspath(f'out/{params.MODEL}_metrics_results_{version}.csv')):
            version += 1

        df = pd.DataFrame().from_dict(metric_dict)
        df.to_csv(ut.abspath(f'out/{params.MODEL}_metrics_results_{version}.csv'), index_label="No")

    @staticmethod
    def __load_model_config(model, conf_type: str | None = None):
        if model == "bert_pretrained":
            config = BertConfig().from_pretrained("Rostlab/prot_bert")
            return config
        elif model == "bert":
            with open(ut.abspath(f'configs/model_configs/{model}_config_default.json')) as f:
                data = json.load(f)
                config = BertConfig(**data)
                return config
        else:
            conf_type = 'default' if conf_type is None else conf_type
            conf_path = ut.abspath(f'configs/model_configs/{model}_config_{conf_type}.json')
            if os.path.exists(conf_path):
                with open(conf_path, 'r') as f:
                    config = json.load(f)
                    return config
            else:
                raise FileNotFoundError("Config file does not exist")

    @staticmethod
    def __unfreeze(layer):
        pass

    @staticmethod
    def __freeze(layer):
        pass
