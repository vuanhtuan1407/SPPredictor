from typing import Any, Literal, Optional

import torch
from torchmetrics import MatthewsCorrCoef, Metric
from torchmetrics.classification import MulticlassMatthewsCorrCoef, BinaryMatthewsCorrCoef, MultilabelMatthewsCorrCoef
from torchmetrics.utilities.enums import ClassificationTask


def _matthews_corrcoef_non_average(confmat: torch.Tensor):
    mcc = []
    tps = torch.diag(confmat)
    fps = torch.sum(confmat, dim=0) - tps
    fns = torch.sum(confmat, dim=1) - tps
    tns = torch.sum(confmat) - (tps + fns + fps)

    for tp, fp, fn, tn in zip(tps, fps, fns, tns):
        numerator = (tp * tn - fp * fn)
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            mcc.append(0)
        else:
            mcc.append((numerator / denominator).item())
    return torch.tensor(mcc, device=confmat.device)


class MulticlassMatthewsCorrCoefNoneAverage(MulticlassMatthewsCorrCoef):
    def compute(self) -> Any:
        return _matthews_corrcoef_non_average(self.confmat)


class MCC(MatthewsCorrCoef):
    def __new__(  # type: ignore[misc]
            cls,
            task: Literal["binary", "multiclass", "multilabel"],
            threshold: float = 0.5,
            num_classes: Optional[int] = None,
            num_labels: Optional[int] = None,
            average: Literal["micro"] | None = 'micro',
            ignore_index: Optional[int] = None,
            validate_args: bool = True,
            **kwargs: Any,
    ) -> Metric:
        """Initialize task and average metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"ignore_index": ignore_index, "validate_args": validate_args})
        if average == "micro":
            if task == ClassificationTask.BINARY:
                return BinaryMatthewsCorrCoef(threshold, **kwargs)
            if task == ClassificationTask.MULTICLASS:
                if not isinstance(num_classes, int):
                    raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
                return MulticlassMatthewsCorrCoef(num_classes, **kwargs)
            if task == ClassificationTask.MULTILABEL:
                if not isinstance(num_labels, int):
                    raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
                return MultilabelMatthewsCorrCoef(num_labels, threshold, **kwargs)
            raise ValueError(f"Not handled value: {task}")
        else:
            # if task == ClassificationTask.BINARY:
            #     return BinaryMatthewsCorrCoefNoneAverage(threshold, **kwargs)
            if task == ClassificationTask.MULTICLASS:
                if not isinstance(num_classes, int):
                    raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
                return MulticlassMatthewsCorrCoefNoneAverage(num_classes, **kwargs)
            # if task == ClassificationTask.MULTILABEL:
            #     if not isinstance(num_labels, int):
            #         raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            #     return MultilabelMatthewsCorrCoefNoneAverage(num_labels, threshold, **kwargs)
            raise ValueError(f"Not handled value: {task}")
