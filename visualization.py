import json
import math

import numpy as np
import pandas as pd
from Bio import SeqIO

import data.data_utils as dut
import params
import utils as ut


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
    # return (pd.DataFrame.from_dict(statistics_on_organisms, orient='index'),
    #         pd.DataFrame.from_dict(statistics_on_labels, orient='index'))
    return statistics_on_organisms, statistics_on_labels


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
    return (pd.DataFrame.from_dict(statistics_on_organisms, orient='index'),
            pd.DataFrame.from_dict(statistics_on_labels, orient='index'))


# TODO: Viet mot class visualize
def _plot_statistics_on_organisms(statistics_on_organisms):
    data = pd.DataFrame(statistics_on_organisms.values(), index=statistics_on_organisms.keys())
    return data.plot(kind='bar', legend="number of samples").get_figure().savefig('statistics_on_organisms.png')


def visualize_data():
    """Visualize how many samples each organism and how many samples each label
    1. visualize the whole dataset
        1.1. read `.fasta`
        1.2. visualize on organisms
        1.3. visualize on labels
    2. visualize training/validation/testing set
        2.1. read `.json`
        2.2. visualize on organisms
        2.3. visualize on labels
    """
    # Part 1
    records = SeqIO.parse(dut.SIGNALP6_PATH, 'fasta')
    statistics_on_organisms, statistics_on_labels = _statistics_on_total(records)

    # Part 2
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
