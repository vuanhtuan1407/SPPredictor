import json
import math

import numpy as np
import pandas as pd
from Bio import SeqIO

import data.data_utils as dut
import params
import utils as ut

PLOT_CONFIG = {
    "kind": "bar",
    "rot": 0.3,
    "save_dir": ut.abspath('out/figures')
}


def _statistics_on_total(records):
    statistics_on_organisms = {}
    statistics_on_labels = {}
    for record in records:
        prot_id, organism, label, partition = str(record.id).split('|')
        if organism not in statistics_on_organisms.keys():
            statistics_on_organisms[organism] = 1
        else:
            statistics_on_organisms[organism] += 1

        if label not in statistics_on_labels.keys():
            statistics_on_labels[label] = 1
        else:
            statistics_on_labels[label] += 1
    return _sort_by_organism(statistics_on_organisms), _sort_by_label(statistics_on_labels)


def _statistics_on_train_val_test(records):
    statistics_on_organisms = {}
    statistics_on_labels = {}
    for record in records:
        organism, label = record['kingdom'], record['label']
        if organism not in statistics_on_organisms.keys():
            statistics_on_organisms[organism] = 1
        else:
            statistics_on_organisms[organism] += 1

        if label not in statistics_on_labels.keys():
            statistics_on_labels[label] = 1
        else:
            statistics_on_labels[label] += 1
    return _sort_by_organism(statistics_on_organisms), _sort_by_label(statistics_on_labels)


def _sort_by_organism(records):
    return dict(sorted(records.items(), key=lambda kv: params.ORGANISMS[kv[0]]))


def _sort_by_label(records):
    return dict(sorted(records.items(), key=lambda kv: params.SP_LABELS[kv[0]]))


def _plot_statistics(statistics: dict, title: str, filename: str = None, save: bool = True):
    data = pd.DataFrame(
        data={"number of samples": statistics.values()},
        index=list(statistics.keys())
    )
    fig = data.plot(kind=PLOT_CONFIG['kind'], title=title, rot=PLOT_CONFIG['rot']).get_figure()
    if filename is not None and save is True:
        fig.savefig(PLOT_CONFIG['save_dir'] + '/' + filename)
    else:
        return fig


def visualize_data():
    """Visualize how many samples each organism and how many samples each label
    """
    # extract all data
    records = SeqIO.parse(dut.SIGNALP6_PATH, 'fasta')
    statistics_on_organisms, statistics_on_labels = _statistics_on_total(records)

    # extract train/val/test set
    train_paths = [f'data/sp_data/train_set_partition_0.json', f'data/sp_data/train_set_partition_1.json']
    val_paths = [f'data/sp_data/test_set_partition_0.json', f'data/sp_data/test_set_partition_1.json']
    test_paths = [f'data/sp_data/train_set_partition_2.json', f'data/sp_data/test_set_partition_2.json']

    train_records, val_records, test_records = [], [], []
    for train_path, val_path, test_path in zip(train_paths, val_paths, test_paths):
        with open(train_path, 'r') as f:
            train_records.extend(json.load(f))
        with open(val_path, 'r') as f:
            val_records.extend(json.load(f))
        with open(test_path, 'r') as f:
            test_records.extend(json.load(f))

    statistics_on_train_organisms, statistics_on_train_labels = _statistics_on_train_val_test(train_records)
    statistics_on_val_organisms, statistics_on_val_labels = _statistics_on_train_val_test(val_records)
    statistics_on_test_organisms, statistics_on_test_labels = _statistics_on_train_val_test(test_records)

    # plot
    _plot_statistics(statistics_on_organisms, title='Statistics on total organisms',
                     filename='statistics_on_organisms.jpg')
    _plot_statistics(statistics_on_labels, title='Statistics on total labels',
                     filename='statistics_on_labels.jpg')

    _plot_statistics(statistics_on_train_organisms, title='Statistics on train organisms',
                     filename='statistics_on_train_organisms.jpg')
    _plot_statistics(statistics_on_train_labels, title='Statistics on train labels',
                     filename='statistics_on_train_labels.jpg')
    _plot_statistics(statistics_on_val_organisms, title='Statistics on val organisms',
                     filename='statistics_on_val_organisms.jpg')
    _plot_statistics(statistics_on_val_labels, title='Statistics on val labels',
                     filename='statistics_on_val_labels.jpg')
    _plot_statistics(statistics_on_test_organisms, title='Statistics on test organisms',
                     filename='statistics_on_test_organisms.jpg')
    _plot_statistics(statistics_on_test_labels, title='Statistics on test labels',
                     filename='statistics_on_test_labels.jpg')


def _metrics_on_organisms(metrics):
    """Metrics include F1, Recall, MCC, Average Precision
    Just read from `.csv` from out/metrics and convert them to pandas DataFrame
    """
    pass


def visualize_metrics():
    """Visualize the result of the metrics on organisms and on labels
    """
    pass


def visualize_results():
    """Visualize the prediction results on organisms and on labels, total, wrong label, correct label
    """
    pass


def read_test_predictions(path):
    data = np.loadtxt(path)
    return data


def plot():
    pass


def visualize():
    statistic = []
    print(f'Below is only the statistic on wrong predictions:\n')
    for kingdom in params.ORGANISMS.keys():
        wrong = [0, 0, 0, 0, 0, 0]
        total = [0, 0, 0, 0, 0, 0]
        predict = read_test_predictions(ut.abspath(f'out/results/{kingdom}_test_prediction_by_transformer.txt'))
        true = read_test_predictions(ut.abspath(f'out/results/{kingdom}_test_true_results.txt'))
        pred_lb = np.argmax(predict, axis=1)
        tmp = 0
        all_lbs = 0
        for true_lb, pred in zip(true, pred_lb):
            all_lbs += 1
            total[math.floor(true_lb)] += 1
            if true_lb == pred:
                tmp += 1
            else:
                wrong[math.floor(true_lb)] += 1
        statistic.append(tmp / all_lbs)
        print(f'{kingdom}: {tmp}/{all_lbs}\n')
        for i in range(0, 6):
            if total[i] != 0:
                print('{}/{} --> {:.1f}\t'.format(wrong[i], total[i], wrong[i] / total[i] * 100))
        print('\n======================================\n')


if __name__ == '__main__':
    visualize_data()
