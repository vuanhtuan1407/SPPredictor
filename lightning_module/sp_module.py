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

import models.model_utils as mut
import params
import tokenizer.tokenizer_utils as tut
import utils as ut


# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
# from torcheval.metrics.functional import multiclass_auroc, multiclass_auprc, multiclass_f1_score


class SPModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        loss_weight = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float)
        self.loss_fn = CrossEntropyLoss(weight=loss_weight)
        self.save_hyperparameters()
        # self.fabric = Fabric()

        # Load config (Remove if unnecessary)
        # self.config = cut.load_config()

        # Tokenizer
        self.tokenizer = tut.load_tokenizer(params.MODEL, params.DATA)

        # Load models
        self.model_type = params.MODEL
        self.model = mut.load_model(params.MODEL, params.DATA, params.CONF_TYPE)

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
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=0.1)
        return optimizer

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any):
        loss.backward()

    def tokenize_input(self, x):
        encoded = self.tokenizer.batch_encode_plus(
            x,
            # max_length=self.model.config['max_len'],
            truncation=True,
            padding=True
        )
        # print(len(encoded['input_ids'][0]))
        return torch.tensor(encoded['input_ids'], dtype=torch.int64, device=self.device)

    def base_step(self, batch, batch_idx):
        x, lb, kingdom = batch
        x = self.tokenize_input(x)
        pred = self.model(x)
        loss = self.loss_fn(pred.float(), lb.float())
        return x, lb, pred, loss, kingdom

    def training_step(self, batch, batch_idx):
        _, _, pred, loss, _ = self.base_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=params.BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        _, lb, pred, loss, _ = self.base_step(batch, batch_idx)
        self.validation_step_outputs_pred.append(pred)
        self.validation_step_outputs_lb.append(lb)
        self.log("val_loss", loss, prog_bar=True, batch_size=params.BATCH_SIZE)
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
            self._save_results_to_txt(all_pred.detach().cpu(), all_lb.detach().cpu(), kingdom=key)

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

        self._save_metrics_to_csv(metric_dict)

    @staticmethod
    def _save_results_to_txt(test_prediction_results, test_true_results, kingdom):
        if not os.path.exists(ut.abspath(f"out/results")):
            os.makedirs(f"out/results", exist_ok=True)

        softmax = Softmax()
        pred_path = f'out/results/{kingdom}_test_prediction_results_by_{params.MODEL}.txt'
        true_path = f'out/results/{kingdom}_test_true_results.txt'

        # print(test_prediction_results)

        np.savetxt(ut.abspath(pred_path), softmax(test_prediction_results), fmt="%.4f")
        # np.savetxt(ut.abspath(true_path), test_true_results, fmt="%d")
        if not os.path.exists(ut.abspath(true_path)):
            np.savetxt(ut.abspath(true_path), test_true_results, fmt="%d")

    @staticmethod
    def _save_metrics_to_csv(metric_dict):
        version = 0
        while os.path.exists(ut.abspath(f'out/metrics/{params.MODEL}_metrics_results_{version}.csv')):
            version += 1

        df = pd.DataFrame().from_dict(metric_dict)
        df.to_csv(ut.abspath(f'out/metrics/{params.MODEL}_metrics_results_{version}.csv'), index_label="No")
