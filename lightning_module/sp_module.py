import os.path
from typing import Any, Dict

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import optim, Tensor
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import Optimizer
from torchmetrics import F1Score, MatthewsCorrCoef, AveragePrecision, Recall

import models.model_utils as mut
import params
import tokenizer.tokenizer_utils as tut
import utils as ut


# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
# from torcheval.metrics.functional import multiclass_auroc, multiclass_auprc, multiclass_f1_score


class SPModule(L.LightningModule):
    def __init__(
            self,
            model_type: str,
            data_type: str,
            conf_type: str = 'default',
            batch_size: int = 8,
            lr: float = 1e-7
    ):
        super().__init__()
        self.save_hyperparameters()

        # Module params
        self.model_type = model_type
        self.data_type = data_type
        self.conf_type = conf_type
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoint_filename = ''

        loss_weight = torch.tensor([0.1, 0.3, 0.5, 0.5, 1, 1], dtype=torch.float)
        self.loss_fn = CrossEntropyLoss(weight=loss_weight)
        # self.fabric = Fabric()

        # Load config (Remove if unnecessary)
        # self.config = cut.load_config()

        # Tokenizer
        self.tokenizer = tut.load_tokenizer(model_type=model_type, data_type=data_type)

        # Load models
        self.model = mut.load_model(model_type=model_type, data_type=data_type, conf_type=conf_type)

        # Load metrics
        self.f1 = F1Score(task='multiclass', num_classes=len(params.SP_LABELS), average=None)
        self.recall = Recall(task="multiclass", num_classes=len(params.SP_LABELS), average=None)
        self.mcc = MatthewsCorrCoef(task='multiclass', num_classes=len(params.SP_LABELS))
        self.average_precision = AveragePrecision(task='multiclass', num_classes=len(params.SP_LABELS), average=None)

        # Outputs from training process
        self.validation_step_outputs_lb = []
        self.validation_step_outputs_pred = []
        self.best_val_loss = 1e6

        self.test_outputs_lb_total = []
        self.test_outputs_pred_total = []
        self.test_outputs_lb_organism = [[]] * len(params.SP_LABELS)
        self.test_outputs_pred_organism = [[]] * len(params.SP_LABELS)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.1)
        return optimizer

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any):
        loss.backward()

    def tokenize_input(self, x):
        encoded = self.tokenizer.batch_encode_plus(
            x,
            # max_length=self.model_type.config['max_len'],
            truncation=True,
            padding=True
        )
        # print(len(encoded['input_ids'][0]))
        return torch.tensor(encoded['input_ids'], dtype=torch.int64, device=self.device)

    def base_step(self, batch, batch_idx):
        x, lb, organism = batch
        x = self.tokenize_input(x)
        pred = self.model(x)
        loss = self.loss_fn(pred.float(), lb.float())
        return x, lb, pred, loss, organism

    def training_step(self, batch, batch_idx):
        _, _, pred, loss, _ = self.base_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        _, lb, pred, loss, _ = self.base_step(batch, batch_idx)
        self.validation_step_outputs_pred.append(pred)
        self.validation_step_outputs_lb.append(lb)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
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
        _, lb, pred, loss, organism = self.base_step(batch, batch_idx)

        # Update outputs for calculate metrics on each class (for total)
        self.test_outputs_pred_total.extend(torch.argmax(pred, dim=1).tolist())
        self.test_outputs_lb_total.extend(torch.argmax(lb, dim=1).tolist())

        # Update outputs for calculate metrics on each class (for each organism)
        for i, o in enumerate(organism):
            self.test_outputs_pred_organism[params.ORGANISMS[o]].append(pred[i].tolist())
            self.test_outputs_lb_organism[params.ORGANISMS[o]].append(lb[i].tolist())

    def on_test_end(self) -> None:
        # TODO: Tạo một metrics dict để lưu các giá trị này lại và print (xem xét tạo một func như class_report của sklearn

        # Apply argmax on these outputs (only for label) and evaluate the metric results
        print(classification_report(self.test_outputs_lb_total, self.test_outputs_pred_total))

        # Calculate metrics on each class (for both on total and on organisms)
        total_index = len(params.ORGANISMS)
        f1_test = [[]] * (total_index + 1)
        recall_test = [[]] * (total_index + 1)
        mcc_test = [[]] * (total_index + 1)
        average_precision_test = [[]] * (total_index + 1)
        for k, o in params.ORGANISMS.items():
            all_pred = torch.tensor(self.test_outputs_pred_organism[o], device=self.device)
            all_lb = torch.tensor(self.test_outputs_lb_organism[o], device=self.device)
            all_lb = torch.argmax(all_lb, dim=1)

            # Print the statistic (the following function has ERROR about syntax)
            # self._save_results_to_txt(all_pred.detach().cpu(), all_lb.detach().cpu(), organism=o)

            self.f1.update(all_pred, all_lb)
            self.recall.update(all_pred, all_lb)
            self.mcc.update(all_pred, all_lb)
            self.average_precision.update(all_pred, all_lb)
            f1_test[o] = (self.f1.compute() * 100).tolist()
            recall_test[o] = (self.recall.compute() * 100).tolist()
            mcc_test[o] = (self.mcc.compute() * 100).tolist()
            average_precision_test[o] = (self.average_precision.compute() * 100).tolist()

            print(
                f'\nMetrics on test set of {k}: '
                f'f1: {f1_test[o]}, '
                f'recall: {recall_test[o]}, '
                f'mcc: {mcc_test[o]}, '
                f'average_precision: {average_precision_test[o]} \n'
            )

            self.f1.reset()
            self.recall.reset()
            self.mcc.reset()
            self.average_precision.reset()

        # all_pred = torch.tensor(self.test_outputs_pred_total, device=self.device)
        # all_lb = torch.tensor(self.test_outputs_lb_total, device=self.device)
        # all_lb = torch.argmax(all_lb, dim=0)
        #
        # # Print the statistic (the following function has ERROR about syntax)
        # # self._save_results_to_txt(all_pred.detach().cpu(), all_lb.detach().cpu(), organism=o)
        #
        # self.f1.update(all_pred, all_lb)
        # self.recall.update(all_pred, all_lb)
        # self.mcc.update(all_pred, all_lb)
        # self.average_precision.update(all_pred, all_lb)
        # f1_test[total_index] = self.f1.compute().tolist()
        # recall_test[total_index] = self.recall.compute().tolist()
        # mcc_test[total_index] = self.mcc.compute().tolist()
        # average_precision_test[total_index] = self.average_precision.compute().tolist()
        #
        # print(
        #     f'\nMetrics on test set of TOTAL: '
        #     f'f1: {f1_test[total_index]}, '
        #     f'recall: {recall_test[total_index]}, '
        #     f'mcc: {mcc_test[total_index]}, '
        #     f'average_precision: {average_precision_test[total_index]} \n'
        # )
        #
        # self.f1.reset()
        # self.recall.reset()
        # self.mcc.reset()
        # self.average_precision.reset()

        metric_dict = {
            "organism": params.ORGANISMS,
            "f1_score": f1_test,
            "recall": recall_test,
            "mcc": mcc_test,
            "average_precision": average_precision_test,
        }

        print(metric_dict)

        self._save_metrics_to_csv(metric_dict)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.checkpoint_filename = checkpoint['best_model_filename'].split('.')[0]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        best_model_path = self.trainer.checkpoint_callback.__getattribute__('best_model_path')
        checkpoint['best_model_filename'] = best_model_path.split('\\')[-1]

    def _save_results_to_txt(self, test_prediction_results, test_true_results, organism):
        if not os.path.exists(ut.abspath(f"out/results")):
            os.makedirs(f"out/results", exist_ok=True)

        softmax = Softmax()
        pred_path = f'out/results/{organism}_test_prediction_by_{self.model_type}.txt'
        true_path = f'out/results/{organism}_test_true.txt'

        np.savetxt(ut.abspath(pred_path), softmax(test_prediction_results), fmt="%.4f")
        # np.savetxt(ut.abspath(true_path), test_true_results, fmt="%d")
        if not os.path.exists(ut.abspath(true_path)):
            np.savetxt(ut.abspath(true_path), test_true_results, fmt="%d")

    def _save_metrics_to_csv(self, metric_dict):
        for k, o in metric_dict['organism'].items():
            metrics_organisms = {
                "f1_score": metric_dict['f1_score'][o],
                "recall": metric_dict['recall'][o],
                "mcc": metric_dict['mcc'][o],
                "average_precision": metric_dict['average_precision'][o],
            }
            df = pd.DataFrame.from_dict(metrics_organisms).transpose().round(2)
            df.to_csv(ut.abspath(f'out/metrics/{self.checkpoint_filename}_test_metrics_{k}.csv'),
                      header=list(params.SP_LABELS.keys()), index_label='metrics')

        # total_index = len(params.ORGANISMS)
        # metrics_total = {
        #     "label": params.SP_LABELS.keys(),
        #     "f1_score": metric_dict['f1_score'][total_index],
        #     "recall": metric_dict['recall'][total_index],
        #     "mcc": metric_dict['mcc'][total_index],
        #     "average_precision": metric_dict['average_precision'][total_index],
        # }
        # df = pd.DataFrame().from_dict(metrics_total)
        # df.to_csv(ut.abspath(f'out/metrics/{self.checkpoint_filename}_test_metrics_TOTAL.csv'), index_label=True)
