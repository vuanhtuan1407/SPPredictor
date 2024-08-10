import os.path
from typing import Any, Dict

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from torch import optim, Tensor
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import Optimizer
from torchmetrics import F1Score, AveragePrecision, Recall

import models.model_utils as mut
import params
import tokenizer.tokenizer_utils as tut
import utils as ut
from metrics import MCC


# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
# from torcheval.metrics.functional import multiclass_auroc, multiclass_auprc, multiclass_f1_score


class SPModule(L.LightningModule):
    def __init__(
            self,
            model_type: str,
            data_type: str,
            conf_type: str = 'default',
            use_organism: bool = False,
            batch_size: int = 8,
            lr: float = 1e-7
    ):
        super().__init__()
        self.save_hyperparameters()

        # Module params
        self.model_type = model_type
        self.data_type = data_type
        self.conf_type = conf_type
        self.use_organism = use_organism
        self.batch_size = batch_size
        self.lr = lr
        if model_type == 'bert' or model_type == 'bert_pretrained':
            self.lr = 1e-5  # according to TSignal, the learning rate for BERT model is fixed to 1e-5
        self.checkpoint_name = ''

        loss_weight = torch.tensor([0.1, 0.3, 0.5, 0.5, 1, 1], dtype=torch.float)
        self.loss_fn = CrossEntropyLoss(weight=loss_weight)
        # self.fabric = Fabric()

        # Load config (Remove if unnecessary)
        # self.config = cut.load_config()

        # Tokenizer
        self.tokenizer = tut.load_tokenizer(model_type=model_type, data_type=data_type)

        # Load models
        self.model = mut.load_model(
            model_type=model_type,
            data_type=data_type,
            conf_type=conf_type,
            use_organism=use_organism
        )

        # Load metrics
        self.f1 = F1Score(task='multiclass', num_classes=len(params.SP_LABELS), average=None)
        self.recall = Recall(task="multiclass", num_classes=len(params.SP_LABELS), average=None)
        self.mcc = MCC(task='multiclass', num_classes=len(params.SP_LABELS), average=None)
        self.average_precision = AveragePrecision(task='multiclass', num_classes=len(params.SP_LABELS), average=None)
        self.macro_ap_score = AveragePrecision(task='multiclass', num_classes=len(params.SP_LABELS), average='macro')
        # self.metrics = MulticlassMetrics(num_classes=len(params.SP_LABELS), average=None, device=self.device)

        # Outputs from training process
        self.validation_outputs_lb = []
        self.validation_outputs_pred = []
        self.best_val_loss = 1e6

        self.test_outputs_lb_total = []
        self.test_outputs_pred_total = []
        self.test_outputs_lb_organism = [[], [], [], [], [], []]
        self.test_outputs_pred_organism = [[], [], [], [], [], []]

    def forward(self, x, organism=None):
        self.model.eval()
        if self.tokenizer is not None:
            x = self.tokenize_input(x)
        if self.use_organism:
            return self.model(x, organism)
        else:
            return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.1)
        return optimizer

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any):
        loss.backward()

    def tokenize_input(self, x):
        # max_length = 0
        if self.model_type == 'bert' or self.model_type == 'bert_pretrained':
            max_length = 70
        else:
            max_length = self.model.config['max_len']
        encoded = self.tokenizer.batch_encode_plus(
            x,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        # encoded = self.tokenizer.batch_encode_plus(
        #     x,
        #     truncation=True,
        #     padding=True
        # )
        # print(len(encoded['input_ids'][0]))
        return torch.tensor(encoded['input_ids'], dtype=torch.int64, device=self.device)

    def base_step(self, batch, batch_idx):
        x, lb, organism = batch
        if self.tokenizer is not None:
            x = self.tokenize_input(x)
        # pred = None  # uncomment this line in case got error do not have variable `pred` defined
        if self.use_organism:
            pred = self.model(x, organism)
        else:
            pred = self.model(x)
        loss = self.loss_fn(pred.float(), lb.float())
        return x, lb, pred, loss, organism

    def training_step(self, batch, batch_idx):
        _, _, pred, loss, _ = self.base_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        _, lb, pred, loss, _ = self.base_step(batch, batch_idx)
        self.validation_outputs_pred.extend(pred.tolist())
        self.validation_outputs_lb.extend(lb.tolist())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        # return loss

    def on_validation_epoch_end(self):
        all_pred = torch.tensor(self.validation_outputs_pred, device=self.device)
        all_lb = torch.tensor(self.validation_outputs_lb, device=self.device)

        val_loss = self.loss_fn(all_pred.float(), all_lb.float())
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        all_lb = torch.argmax(all_lb, dim=1)

        self.f1.update(all_pred, all_lb)
        self.recall.update(all_pred, all_lb)
        self.mcc.update(all_pred, all_lb)
        self.average_precision.update(all_pred, all_lb)

        print(
            f"\nMetrics on validation set: "
            f"best_val_loss: {self.best_val_loss}, "
            f"f1: {self.f1.compute()}, "
            f"recall: {self.recall.compute()}, "
            f"mcc: {self.mcc.compute()}, "
            f"average_precision: {self.average_precision.compute()} \n"
        )

        self.validation_outputs_lb.clear()
        self.validation_outputs_pred.clear()
        self.f1.reset()
        self.recall.reset()
        self.mcc.reset()
        self.average_precision.reset()

    def test_step(self, batch, batch_idx):
        _, lb, pred, loss, organism = self.base_step(batch, batch_idx)

        # detach prediction results from pytorch graph in order to decrease memory usage
        pred = pred.clone().detach()

        # Update outputs for calculate metrics on each class (for total)
        self.test_outputs_pred_total.extend(pred.tolist())
        self.test_outputs_lb_total.extend(lb.tolist())

        # Update outputs for calculate metrics on each class (for each organism)
        for i, o in enumerate(organism):
            self.test_outputs_pred_organism[o].append(pred[i].tolist())
            self.test_outputs_lb_organism[o].append(lb[i].tolist())

    def on_test_end(self) -> None:
        # TODO: Tạo một metrics dict để lưu các giá trị này lại và print (xem xét tạo một func như class_report của sklearn

        softmax = Softmax(dim=-1)

        # Apply argmax on these outputs (only for label) and evaluate the metric results
        total_pred = torch.tensor(self.test_outputs_pred_total, device=self.device)
        total_lb = torch.tensor(self.test_outputs_lb_total, device=self.device)
        print(classification_report(torch.argmax(total_pred, dim=1).tolist(), torch.argmax(total_lb, dim=1).tolist(),
                                    zero_division=0))

        # macro_ap = average_precision_score(torch.argmax(total_lb, dim=1).tolist(), total_pred.tolist(), average='macro')
        macro_ap = (self.macro_ap_score(total_pred, torch.argmax(total_lb, dim=1)) * 100).item()
        micro_ap = average_precision_score(torch.argmax(total_lb, dim=1).tolist(), total_pred.tolist(), average='micro')

        # Calculate metrics on each class (for both on total and on organisms)
        total_index = len(params.ORGANISMS)
        f1_test = [[], [], [], [], []]
        recall_test = [[], [], [], [], []]
        mcc_test = [[], [], [], [], []]
        average_precision_test = [[], [], [], [], []]
        macro_ap_orgs = [[], [], [], []]
        for k, o in params.ORGANISMS.items():
            all_pred = softmax(torch.tensor(self.test_outputs_pred_organism[o], device=self.device))
            all_lb = torch.tensor(self.test_outputs_lb_organism[o], device=self.device)
            all_lb = torch.argmax(all_lb, dim=1)

            macro_ap_orgs[o] = (self.macro_ap_score(all_pred, all_lb) * 100).item()

            # Print the statistic (the following function has ERROR about syntax)
            if params.USE_LOGGER:
                self._save_results_to_txt(all_pred.clone().detach().cpu(), all_lb.clone().detach().cpu(), organism=k)

            f1_test[o] = (self.f1(all_pred, all_lb) * 100).tolist()
            recall_test[o] = (self.recall(all_pred, all_lb) * 100).tolist()
            mcc_test[o] = (self.mcc(all_pred, all_lb) * 100).tolist()
            average_precision_test[o] = (self.average_precision(all_pred, all_lb) * 100).tolist()

            # print(
            #     f'\nMetrics on test set of {k}: '
            #     f'f1: {f1_test[o]}, '
            #     f'recall: {recall_test[o]}, '
            #     f'mcc: {mcc_test[o]}, '
            #     f'average_precision: {average_precision_test[o]} \n'
            # )

            self.f1.reset()
            self.recall.reset()
            self.mcc.reset()
            self.average_precision.reset()

        all_pred = total_pred
        all_lb = total_lb
        all_lb = torch.argmax(all_lb, dim=1)

        self.f1.update(all_pred, all_lb)
        self.recall.update(all_pred, all_lb)
        self.mcc.update(all_pred, all_lb)
        self.average_precision.update(all_pred, all_lb)
        f1_test[total_index] = (self.f1.compute() * 100).tolist()
        recall_test[total_index] = (self.recall.compute() * 100).tolist()
        mcc_test[total_index] = (self.mcc.compute() * 100).tolist()
        average_precision_test[total_index] = (self.average_precision.compute() * 100).tolist()

        # print(
        #     f'\nMetrics on test set of TOTAL: '
        #     f'f1: {f1_test[total_index]}, '
        #     f'recall: {recall_test[total_index]}, '
        #     f'mcc: {mcc_test[total_index]}, '
        #     f'average_precision: {average_precision_test[total_index]} \n'
        # )

        self.f1.reset()
        self.recall.reset()
        self.mcc.reset()
        self.average_precision.reset()

        metric_dict = {
            "f1_score": f1_test,
            "recall": recall_test,
            "mcc": mcc_test,
            "average_precision": average_precision_test,
        }

        # print(metric_dict)

        print(macro_ap, micro_ap, macro_ap_orgs)

        if params.USE_LOGGER:
            self._save_ap_score_to_csv(macro_ap, micro_ap, macro_ap_orgs)
            self._save_metrics_to_csv(metric_dict)

    def predict_step(self, batch, batch_idx):
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.checkpoint_name = params.CHECKPOINT.split('.')[0]

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
        for k, o in params.ORGANISMS.items():
            metrics_organisms = {
                "f1_score": metric_dict['f1_score'][o],
                "recall": metric_dict['recall'][o],
                "mcc": metric_dict['mcc'][o],
                "average_precision": metric_dict['average_precision'][o],
            }
            df = pd.DataFrame.from_dict(metrics_organisms).transpose().round(2)
            df.to_csv(ut.abspath(f'out/metrics/{self.checkpoint_name}_test_{k}.csv'),
                      header=list(params.SP_LABELS.keys()), index_label='metrics', na_rep=str(0.0))

        total_index = len(params.ORGANISMS)
        metrics_total = {
            "f1_score": metric_dict['f1_score'][total_index],
            "recall": metric_dict['recall'][total_index],
            "mcc": metric_dict['mcc'][total_index],
            "average_precision": metric_dict['average_precision'][total_index],
        }
        df = pd.DataFrame().from_dict(metrics_total).transpose().round(2)
        df.to_csv(ut.abspath(f'out/metrics/{self.checkpoint_name}_test_metrics_TOTAL.csv'),
                  header=list(params.SP_LABELS.keys()), index_label='metrics', na_rep=str(0.0))

    def _save_ap_score_to_csv(self, macro_ap, micro_ap, macro_ap_orgs):
        data = self.data_type
        if data == "aa":
            data = "AA Seq"
        elif data == "smiles":
            data = "SMILES"
        elif data == "graph":
            data = "Graph 3D"
        model = f"{self.model_type.upper()}, {data}, Organism: {'Yes' if self.use_organism else 'No'}"

        models = []
        organisms = []
        for k in params.ORGANISMS.keys():
            models.append(model)
            organisms.append(k)

        ap_score_orgs = {
            "model": models,
            "organism": organisms,
            "macro_ap": macro_ap_orgs
        }

        df_orgs = pd.DataFrame(ap_score_orgs)
        df_orgs.to_csv(ut.abspath(f'out/metrics/{self.checkpoint_name}_ap_score_ORG.csv'), index=True,
                       index_label="index", na_rep=str(0.0))

        ap_score = {
            "model": [model],
            "macro_ap": [macro_ap],
            "micro_ap": [micro_ap]
        }

        df = pd.DataFrame(ap_score)
        df.to_csv(ut.abspath(f'out/metrics/{self.checkpoint_name}_ap_score_TOTAL.csv'), index=True, index_label="index",
                  na_rep=str(0.0))
