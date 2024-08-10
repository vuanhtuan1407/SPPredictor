import json
import math

import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib import pyplot as plt

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
    statistics_on_labels_organism = {}

    for record in records:
        _, organism, label, _ = str(record.id).split('|')
        if organism not in statistics_on_organisms.keys():
            statistics_on_organisms[organism] = 1
        else:
            statistics_on_organisms[organism] += 1

        if label not in statistics_on_labels.keys():
            statistics_on_labels[label] = 1
        else:
            statistics_on_labels[label] += 1

        if organism not in statistics_on_labels_organism.keys():
            statistics_on_labels_organism[organism] = {}
            statistics_on_labels_organism[organism][label] = 1
        else:
            if label not in statistics_on_labels_organism[organism].keys():
                statistics_on_labels_organism[organism][label] = 1
            else:
                statistics_on_labels_organism[organism][label] += 1

    for o, lb in statistics_on_labels_organism.items():
        statistics_on_labels_organism[o] = _sort_by_label(statistics_on_labels_organism[o])

    return _sort_by_organism(statistics_on_organisms), _sort_by_label(statistics_on_labels), _sort_by_organism(
        statistics_on_labels_organism)


def _statistics_on_train_val_test(records):
    statistics_on_organisms = {}
    statistics_on_labels = {}
    statistics_on_labels_organism = {}

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

    for record in records:
        organism, label = record['kingdom'], record['label']
        if organism not in statistics_on_labels_organism.keys():
            statistics_on_labels_organism[organism] = {}
            statistics_on_labels_organism[organism][label] = 1
        else:
            if label not in statistics_on_labels_organism[organism].keys():
                statistics_on_labels_organism[organism][label] = 1
            else:
                statistics_on_labels_organism[organism][label] += 1
    for o, lb in statistics_on_labels_organism.items():
        statistics_on_labels_organism[o] = _sort_by_label(statistics_on_labels_organism[o])

    return _sort_by_organism(statistics_on_organisms), _sort_by_label(statistics_on_labels), _sort_by_organism(
        statistics_on_labels_organism)


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
        data.transpose().to_csv(ut.abspath(f'out/metrics/{filename.split(".")[0]}.csv'), index_label='labels')
        fig.savefig(PLOT_CONFIG['save_dir'] + '/' + filename)
        plt.close(fig)
    else:
        return fig


def visualize_data():
    """Visualize how many samples each organism and how many samples each label
    """
    # extract all data
    records = SeqIO.parse(dut.SIGNALP6_PATH, 'fasta')
    statistics_on_organisms, statistics_on_labels, statistics_on_labels_organism = _statistics_on_total(records)

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

    statistics_on_train_organisms, statistics_on_train_labels, statistics_on_train_labels_organism = _statistics_on_train_val_test(
        train_records)
    statistics_on_val_organisms, statistics_on_val_labels, statistics_on_val_labels_organism = _statistics_on_train_val_test(
        val_records)
    statistics_on_test_organisms, statistics_on_test_labels, statistics_on_test_labels_organism = _statistics_on_train_val_test(
        test_records)

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

    for k in params.ORGANISMS.keys():
        _plot_statistics(statistics_on_labels_organism[k], title=f'Statistics on total labels {k}',
                         filename=f'statistics_on_labels_{k}.jpg')

        _plot_statistics(statistics_on_train_labels_organism[k], title=f'Statistics on train labels {k}',
                         filename=f'statistics_on_train_labels_{k}.jpg')
        _plot_statistics(statistics_on_val_labels_organism[k], title=f'Statistics on val labels {k}',
                         filename=f'statistics_on_val_labels_{k}.jpg')
        _plot_statistics(statistics_on_test_labels_organism[k], title=f'Statistics on test labels {k}',
                         filename=f'statistics_on_test_labels_{k}.jpg')


def statistic_num_samples_to_csv():
    records = SeqIO.parse(dut.SIGNALP6_PATH, 'fasta')
    statistics_on_organisms, statistics_on_labels, statistics_on_labels_organism = _statistics_on_total(records)

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

    statistics_on_train_organisms, statistics_on_train_labels, statistics_on_train_labels_organism = _statistics_on_train_val_test(
        train_records)
    statistics_on_val_organisms, statistics_on_val_labels, statistics_on_val_labels_organism = _statistics_on_train_val_test(
        val_records)
    statistics_on_test_organisms, statistics_on_test_labels, statistics_on_test_labels_organism = _statistics_on_train_val_test(
        test_records)


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


def _extract_metric_filename(filename):
    """Extract info from metric filename: model, data, used_org, organism
    """
    model, data, conf = filename.split('-')[0:3]
    use_org = int(filename.split("-")[3].split('_')[0])
    organism = filename.split("-")[3].split('_')[-1].split('.')[0]

    return model, data, conf, use_org, organism


def _extract_model_filename(filename):
    model, data, conf = filename.split('-')[0:3]
    use_org = int(filename.split("-")[3].split('_')[0])
    is_freeze = int(filename.split('-')[3].split('_')[-1] == 'v1')
    return model, data, conf, use_org, is_freeze


def visualize_metrics(checkpoint_names: list[str]):
    """Visualize the result of the metrics on organisms and on labels
    """
    legends = []
    checkpoint_names = checkpoint_names
    dfs = []
    for k, o in params.ORGANISMS.items():
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(k)
        metrics = {
            # "f1_score": {},
            # "recall": {},
            # "mcc": {},
            "average_precision": {}
        }
        for checkpoint_name in checkpoint_names:
            metric_filename = checkpoint_name + f"_test_{k}.csv"
            model, data, _, used_org, _ = _extract_metric_filename(metric_filename)
            metric_path = ut.abspath(f'out/metrics/{metric_filename}')
            dfs.append(pd.read_csv(metric_path))
            if used_org:
                legends.append(f"{model}, use org, {data}")
            else:
                legends.append(f"{model}, no org, {data}")

        for i, metric in enumerate(metrics.keys()):
            for j, model in enumerate(legends):
                df = dfs[j]
                metrics[metric][model] = df[df['metrics'] == metric].values[0][1:]
            ax = plt.subplot(1, 1, i + 1)
            bar_plot(ax, metrics[metric], total_width=0.8, single_width=0.9)
            plt.xticks(range(6), list(params.SP_LABELS.keys()))
            plt.title(metric)
        # plt.show()
        fig.savefig(ut.abspath(f'out/figures/metrics_on_{k}.png'))

    fig = plt.figure(figsize=(7, 6))
    fig.suptitle("TOTAL")
    metrics = {
        # "f1_score": {},
        # "recall": {},
        # "mcc": {},
        "average_precision": {}
    }
    for checkpoint_name in checkpoint_names:
        metric_filename = checkpoint_name + f"_test_metrics_TOTAL.csv"
        model, data, _, used_org, _ = _extract_metric_filename(metric_filename)
        metric_path = ut.abspath(f'out/metrics/{metric_filename}')
        dfs.append(pd.read_csv(metric_path))
        if used_org:
            legends.append(f"{model}, use org, {data}")
        else:
            legends.append(f"{model}, no org, {data}")

    for i, metric in enumerate(metrics.keys()):
        for j, model in enumerate(legends):
            df = dfs[j]
            metrics[metric][model] = df[df['metrics'] == metric].values[0][1:]
        ax = plt.subplot(1, 1, i + 1)
        bar_plot(ax, metrics[metric], total_width=0.8, single_width=0.9)
        plt.xticks(range(6), list(params.SP_LABELS.keys()))
        plt.title(metric)
    # plt.show()
    fig.savefig(ut.abspath(f'out/figures/metrics_on_TOTAL.png'))


def select_and_save_ap_score_to_csv(models: list[str]):
    filenames = []
    model_types = []
    # data_types = []
    # use_orgs = []
    organisms = []
    ap_score = []
    NO_SP, SP, LIPO, TAT, PILIN, TATLIPO = [], [], [], [], [], []
    for model in models:
        model_type, data_type, conf, use_org, is_freeze = _extract_model_filename(model)
        if is_freeze:
            model_type = model_type + "_freezed"
        if data_type == 'aa':
            data_type = "AA Seq"
        if data_type == 'smiles':
            data_type = "SMILES"
        if data_type == 'graph':
            data_type = "Graph 3D"
        model_type = f"{model_type.upper()}, {data_type}, Organism: {'Yes' if use_org else 'No'}"
        for org in params.ORGANISMS.keys():
            model_types.append(model_type)
            organisms.append(org)
            filename = f"{model}_test_{org}.csv"
            filenames.append(filename)
    for filename in filenames:
        df = pd.read_csv(ut.abspath(f'out/metrics/{filename}'))
        ap = df[df['metrics'] == 'average_precision'].values[0][1:].tolist()
        NO_SP.append(ap[0])
        SP.append(ap[1])
        LIPO.append(ap[2])
        TAT.append(ap[3])
        PILIN.append(ap[4])
        TATLIPO.append(ap[5])
        # ap_score.append(df[df['metrics'] == 'average_precision'].values[0][1:].tolist())

    df_data = {
        "model_type": model_types,
        "organism": organisms,
        "NO_SP": NO_SP,
        "SP": SP,
        "LIPO": LIPO,
        "TAT": TAT,
        "PILIN": PILIN,
        "TATLIPO": TATLIPO
    }
    df = pd.DataFrame(df_data)
    df.to_csv(ut.abspath(f'out/metrics/ap_score.csv'), index=True, index_label='index')

    filenames = []
    model_types = []
    # data_types = []
    # use_orgs = []
    organisms = []
    ap_score = []
    NO_SP, SP, LIPO, TAT, PILIN, TATLIPO = [], [], [], [], [], []
    for model in models:
        model_type, data_type, conf, use_org, is_freeze = _extract_model_filename(model)
        if is_freeze:
            model_type = model_type + "_freezed"
        if data_type == 'aa':
            data_type = "AA seq"
        if data_type == 'smiles':
            data_type = "SMILES"
        if data_type == 'graph':
            data_type = "graph 3D"
        model_type = f"{model_type.upper()}, {data_type}, Organism: {'Yes' if use_org else 'No'}"
        model_types.append(model_type)
        organisms.append('TOTAL')
        filenames.append(f'{model}_test_metrics_TOTAL.csv')
    for filename in filenames:
        df = pd.read_csv(ut.abspath(f'out/metrics/{filename}'))
        ap = df[df['metrics'] == 'average_precision'].values[0][1:].tolist()
        NO_SP.append(ap[0])
        SP.append(ap[1])
        LIPO.append(ap[2])
        TAT.append(ap[3])
        PILIN.append(ap[4])
        TATLIPO.append(ap[5])
        # ap_score.append(df[df['metrics'] == 'average_precision'].values[0][1:].tolist())

    df_data = {
        "model_type": model_types,
        "organism": organisms,
        "NO_SP": NO_SP,
        "SP": SP,
        "LIPO": LIPO,
        "TAT": TAT,
        "PILIN": PILIN,
        "TATLIPO": TATLIPO
    }
    df = pd.DataFrame(df_data)
    df.to_csv(ut.abspath(f'out/metrics/ap_score_TOTAL.csv'), index=True, index_label='index')

    filenames = []
    model_types = []
    # data_types = []
    # use_orgs = []
    labels = []
    label_ids = []
    ap_score = []
    NO_SP, SP, LIPO, TAT, PILIN, TATLIPO = [], [], [], [], [], []
    for model in models:
        filenames.append(f'{model}_test_metrics_TOTAL.csv')
    for i, filename in enumerate(filenames):
        model = models[i]
        model_type, data_type, conf, use_org, is_freeze = _extract_model_filename(model)
        if is_freeze:
            model_type = model_type + "_freezed"
        if data_type == 'aa':
            data_type = "AA seq"
        if data_type == 'smiles':
            data_type = "SMILES"
        if data_type == 'graph':
            data_type = "graph 3D"
        model_type = f"{model_type.upper()}, {data_type}, Organism: {'Yes' if use_org else 'No'}"

        df = pd.read_csv(ut.abspath(f'out/metrics/{filename}'))
        ap = df[df['metrics'] == 'average_precision'].values[0][1:].tolist()
        for k, lb in params.SP_LABELS.items():
            model_types.append(model_type)
            labels.append(k)
            label_ids.append(lb)
            ap_score.append(ap[lb])
        # ap_score.append(df[df['metrics'] == 'average_precision'].values[0][1:].tolist())

    df_data = {
        "model_type": model_types,
        "labels": labels,
        "label_ids": label_ids,
        "ap_score": ap_score
    }
    df = pd.DataFrame(df_data)
    df.to_csv(ut.abspath(f'out/metrics/ap_score_TOTAL_visualize.csv'), index=True, index_label="index")


def select_and_save_macro_micro_ap_score_to_csv(models):
    # total
    model_types = [[], []]
    orgs = []
    macro_ap = [[], []]
    micro_ap = []
    for model in models:
        ap_org = pd.read_csv(ut.abspath(f'out/metrics/{model}_ap_score_ORG.csv'))
        ap_total = pd.read_csv(ut.abspath(f'out/metrics/{model}_ap_score_TOTAL.csv'))

        # ORG
        for k, o in params.ORGANISMS.items():
            if '_v1' in model:
                model_type, data, use_org = ap_org[ap_org["index"] == o]['model'].values[0].split(',')
                model_type = model_type + "_freeze"
                model_types[0].append(f'{model_type},{data},{use_org}')
            else:
                model_types[0].append(ap_org[ap_org['index'] == o]['model'].values[0])
            orgs.append(k)
            macro_ap[0].append(ap_org[ap_org["index"] == o]['macro_ap'].values[0])

        if '_v1' in model:
            model_type, data, use_org = ap_org[ap_org["index"] == 0]['model'].values[0].split(',')
            model_type = model_type + "_freeze"
            model_types[1].append(f'{model_type},{data},{use_org}')
        else:
            model_types[1].append(ap_org[ap_org['index'] == 0]['model'].values[0])
        macro_ap[1].append(ap_total[ap_total["index"] == 0]['macro_ap'].values[0])
        micro_ap.append(ap_total[ap_total["index"] == 0]['micro_ap'].values[0])

    metrics_org = {
        "model": model_types[0],
        "organism": orgs,
        "macro_ap": [f'{(m/100):.3f}' for m in macro_ap[0]],
    }

    df_org = pd.DataFrame(metrics_org)
    df_org.to_csv(ut.abspath(f'out/metrics/ap_score_combine_ORG.csv'), index=True, index_label="index")

    metrics_total = {
        "model": model_types[1],
        "macro_ap": [f'{(m/100):.3f}' for m in macro_ap[1]],
        "micro_ap": [f'{m :.3f}' for m in micro_ap]
    }

    df_total = pd.DataFrame(metrics_total)
    df_total.to_csv(ut.abspath(f'out/metrics/ap_score_combine_TOTAL.csv'), index=True, index_label="index")


def statistic_on_ap_score():
    df = pd.read_csv(ut.abspath(f'out/metrics/ap_score.csv'))
    df_total = pd.read_csv(ut.abspath(f'out/metrics/ap_score_TOTAL.csv'))

    models = df_total['model_type'].unique()
    num_models = len(models)

    ranking_table = np.zeros((len(params.ORGANISMS), num_models, num_models))
    ranking_table_total = np.zeros((num_models, num_models))

    # for TOTAL
    for k, l in params.SP_LABELS.items():
        ranks = np.array(df_total[k].rank(method='min', ascending=False), dtype=int)
        for i, rank in enumerate(ranks):
            ranking_table_total[i][rank - 1] += 1

    for k1, o in params.ORGANISMS.items():
        for k2, l in params.SP_LABELS.items():
            if df[df['organism'] == k1][k2].sum() != 0.0:
                ranks = np.array(df[df['organism'] == k1][k2].rank(method='min', ascending=False), dtype=int)
                for i, rank in enumerate(ranks):
                    ranking_table[o][i][rank - 1] += 1

    model_types = []
    ranks = []
    counts = []

    for i in range(num_models):
        for rank in range(num_models):
            model_types.append(models[i])
            ranks.append(rank + 1)
            counts.append(ranking_table_total[i, rank])

    df_data = {
        "model_type": model_types,
        "ranks": ranks,
        "counts": counts
    }

    df_save = pd.DataFrame(df_data)
    df_save.to_csv(ut.abspath(f'out/metrics/ranking_TOTAL_visualize.csv'), index=True, index_label='index')

    model_types = []
    ranks = []
    counts = []
    organism = []

    for k, o in params.ORGANISMS.items():
        for i in range(num_models):
            for rank in range(num_models):
                model_types.append(models[i])
                ranks.append(rank + 1)
                organism.append(k)
                counts.append(ranking_table[o, i, rank])

    df_data = {
        "model_type": model_types,
        "organism": organism,
        "ranks": ranks,
        "counts": counts
    }

    df_save = pd.DataFrame(df_data)
    df_save.to_csv(ut.abspath(f'out/metrics/ranking_visualize.csv'), index=True, index_label='index')


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
    # visualize_data()
    visualize_metrics([
        "cnn-aa-default-0_epochs=100",
        # "transformer-aa-lite-0_epochs=100",
        # "transformer-aa-lite-1_epochs=100",
        "transformer-aa-lite-0_epochs=100",
        "lstm-aa-default-0_epochs=100",
        "st_bilstm-aa-default-0_epochs=100",
        "bert-aa-default-0_epochs=100",
        "bert_pretrained-aa-default-0_epochs=100",
        "bert_pretrained-aa-default-0_epochs=100_v1",
        "cnn_trans-aa-lite-0_epochs=100",
        "cnn-smiles-default-0_epochs=100",
        "transformer-smiles-lite-0_epochs=100",
        "cnn_trans-smiles-lite-0_epochs=100",
        "gconv-graph-heavy-0_epochs=100",
        "gconv_trans-graph-default-0_epochs=100",
        "cnn-aa-default-1_epochs=100",
        "transformer-aa-lite-1_epochs=100",
        "lstm-aa-lite-1_epochs=100",
        "st_bilstm-aa-lite-1_epochs=100",
        "bert-aa-default-1_epochs=100",
        "bert_pretrained-aa-default-1_epochs=100",
        "bert_pretrained-aa-default-1_epochs=100_v1",
        "cnn_trans-aa-lite-1_epochs=100",
        "cnn-smiles-default-1_epochs=100",
        "transformer-smiles-lite-1_epochs=100",
        "cnn_trans-smiles-lite-1_epochs=100",
        "gconv-graph-heavy-1_epochs=100",
        "gconv_trans-graph-default-1_epochs=100",
    ])

    select_and_save_ap_score_to_csv([
        "cnn-aa-default-0_epochs=100",
        # "transformer-aa-lite-0_epochs=100",
        # "transformer-aa-lite-1_epochs=100",
        "transformer-aa-lite-0_epochs=100",
        "lstm-aa-default-0_epochs=100",
        "st_bilstm-aa-default-0_epochs=100",
        "bert-aa-default-0_epochs=100",
        "bert_pretrained-aa-default-0_epochs=100",
        "bert_pretrained-aa-default-0_epochs=100_v1",
        "cnn_trans-aa-lite-0_epochs=100",
        "cnn-smiles-default-0_epochs=100",
        "transformer-smiles-lite-0_epochs=100",
        "cnn_trans-smiles-lite-0_epochs=100",
        "gconv-graph-heavy-0_epochs=100",
        "gconv_trans-graph-default-0_epochs=100",
        "cnn-aa-default-1_epochs=100",
        "transformer-aa-lite-1_epochs=100",
        "lstm-aa-lite-1_epochs=100",
        "st_bilstm-aa-lite-1_epochs=100",
        "bert-aa-default-1_epochs=100",
        "bert_pretrained-aa-default-1_epochs=100",
        "bert_pretrained-aa-default-1_epochs=100_v1",
        "cnn_trans-aa-lite-1_epochs=100",
        "cnn-smiles-default-1_epochs=100",
        "transformer-smiles-lite-1_epochs=100",
        "cnn_trans-smiles-lite-1_epochs=100",
        "gconv-graph-heavy-1_epochs=100",
        "gconv_trans-graph-default-1_epochs=100",

    ])

    statistic_on_ap_score()

    select_and_save_macro_micro_ap_score_to_csv([
        "cnn-aa-default-0_epochs=100",
        # "transformer-aa-lite-0_epochs=100",
        # "transformer-aa-lite-1_epochs=100",
        "transformer-aa-lite-0_epochs=100",
        "lstm-aa-default-0_epochs=100",
        "st_bilstm-aa-default-0_epochs=100",
        "bert-aa-default-0_epochs=100",
        "bert_pretrained-aa-default-0_epochs=100",
        "bert_pretrained-aa-default-0_epochs=100_v1",
        "cnn_trans-aa-lite-0_epochs=100",
        "cnn-smiles-default-0_epochs=100",
        "transformer-smiles-lite-0_epochs=100",
        "cnn_trans-smiles-lite-0_epochs=100",
        "gconv-graph-heavy-0_epochs=100",
        "gconv_trans-graph-default-0_epochs=100",
        "cnn-aa-default-1_epochs=100",
        "transformer-aa-lite-1_epochs=100",
        "lstm-aa-lite-1_epochs=100",
        "st_bilstm-aa-lite-1_epochs=100",
        "bert-aa-default-1_epochs=100",
        "bert_pretrained-aa-default-1_epochs=100",
        "bert_pretrained-aa-default-1_epochs=100_v1",
        "cnn_trans-aa-lite-1_epochs=100",
        "cnn-smiles-default-1_epochs=100",
        "transformer-smiles-lite-1_epochs=100",
        "cnn_trans-smiles-lite-1_epochs=100",
        "gconv-graph-heavy-1_epochs=100",
        "gconv_trans-graph-default-1_epochs=100",
    ])
