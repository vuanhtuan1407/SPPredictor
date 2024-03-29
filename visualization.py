import math
from pathlib import Path

import numpy as np

import params


def read_test_predictions(path):
    data = np.loadtxt(path)
    return data


def plot():
    pass


def visualize():
    statistic = []
    for kingdom in params.KINGDOM.keys():
        wrong = [0, 0, 0, 0, 0, 0]
        total = [0, 0, 0, 0, 0, 0]
        predict = read_test_predictions(str(Path(params.ROOT_DIR) / f'out/{kingdom}_test_prediction_results_by_cnn.txt'))
        true = read_test_predictions(str(Path(params.ROOT_DIR) / f'out/{kingdom}_true_prediction_results.txt'))
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
    visualize()
