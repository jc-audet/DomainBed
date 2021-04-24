# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys
import time
import torch

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

import matplotlib.pyplot as plt 


def get_grouped_anneal(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        group = (r['args']['trial_seed'],
                r['args']['anneal_iter'],
                r["args"]["dataset"],
                r["args"]["algorithm"])
        result[group].append(r)
    return Q([{"Trial_seed": s, "Anneal_iter": t, "dataset": d, "algorithm": a,
        "records": Q(r)} for (s, t, d,a),r in result.items()])


def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm", 
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

def plot_anneal_experiment(records):

    grouped_records = get_grouped_anneal(records)

    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    anneal_order = []
    best_accs = collections.defaultdict(lambda: [])

    for exp in grouped_records:

        best_test_acc = (exp['records'][-1]['env0_in_acc'] + exp['records'][-1]['env0_out_acc'] ) / 2.

        id = (exp['Anneal_iter'],
            exp['algorithm'],
            exp['dataset'])

        best_accs[id].append(best_test_acc)
    
    best_accs = {k: np.mean(v) for k, v in best_accs.items()}
    
    anneal_iter = collections.defaultdict(lambda: [])
    best_anneal = collections.defaultdict(lambda: [])

    for (t,a,d), v in best_accs.items():
        anneal_iter[(a,d)].append(t)
        best_anneal[(a,d)].append(v)
    
    for k, v in best_anneal.items():
        sort = np.argsort(anneal_iter[k])
        anneal_iter[k] = np.array(anneal_iter[k])[sort]
        best_anneal[k] = np.array(best_anneal[k])[sort]
        
    for d in dataset_names:
        plt.figure()
        for a in alg_names:
            plt.plot(anneal_iter[(a,d)], best_anneal[(a,d)], label=a)
        plt.title(d)
        plt.legend()

    plt.show()

def plot_anneal_experiment_max(records):

    grouped_records = get_grouped_anneal(records)

    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    anneal_order = []
    best_accs = collections.defaultdict(lambda: [])

    for exp in grouped_records:

        best_test_acc = 0
        for it in range(10, len(exp['records'])):
            
            test_acc = (exp['records'][it]['env0_in_acc'] + exp['records'][it]['env0_out_acc'] ) / 2.

            if test_acc > best_test_acc:
                best_test_acc = test_acc

        id = (exp['Anneal_iter'],
            exp['algorithm'],
            exp['dataset'])

        best_accs[id].append(best_test_acc)
    
    best_accs = {k: np.mean(v) for k, v in best_accs.items()}
    
    anneal_iter = collections.defaultdict(lambda: [])
    best_anneal = collections.defaultdict(lambda: [])

    for (t,a,d), v in best_accs.items():
        anneal_iter[(a,d)].append(t)
        best_anneal[(a,d)].append(v)
    
    for k, v in best_anneal.items():
        sort = np.argsort(anneal_iter[k])
        anneal_iter[k] = np.array(anneal_iter[k])[sort]
        best_anneal[k] = np.array(best_anneal[k])[sort]
        
    for d in dataset_names:
        plt.figure()
        for a in alg_names:
            plt.plot(anneal_iter[(a,d)], best_anneal[(a,d)], label=a)
        plt.title(d)
        plt.legend()

    plt.show()


def plot_training_curve(records, algorithm):

    grouped_records = get_grouped_anneal(records)

    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in alg_names if n==algorithm])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    anneal_order = []
    test_accs = collections.defaultdict(lambda: [])
    train_accs = collections.defaultdict(lambda: [])
    val_accs = collections.defaultdict(lambda: [])
    reg_values = collections.defaultdict(lambda: [])
    for exp in grouped_records:

        id = (exp['Anneal_iter'],
            exp['algorithm'],
            exp['dataset'])
        test_accs[id].append([])
        train_accs[id].append([])
        val_accs[id].append([])
        reg_values[id].append([])
        
        for it in range(len(exp['records'])):

            it_test_acc = (exp['records'][it]['env0_in_acc'] + exp['records'][it]['env0_out_acc'] ) / 2.
            it_train_acc = (exp['records'][it]['env1_in_acc'] + exp['records'][it]['env2_in_acc'] ) / 2.
            it_val_acc = (exp['records'][it]['env1_out_acc'] + exp['records'][it]['env2_out_acc'] ) / 2.
            test_accs[id][-1].append(it_test_acc)
            train_accs[id][-1].append(it_train_acc)
            val_accs[id][-1].append(it_val_acc)

            if id[1] == 'IGA':
                reg_values[id][-1].append(exp['records'][it]['penalty'])
    
    for d in dataset_names:
        for id, v in test_accs.items():
            if id[1]==algorithm:
                # Test accs
                plt.figure()
                for s in range(len(v)):
                    plt.plot([10*i for i in range(201)], test_accs[id][s])
                plt.title(id)
                plt.axvline(id[0], color='k', linestyle='--')
                plt.ylabel('Test acc')

                # acc gap
                plt.figure()
                for s in range(len(v)):
                    plt.plot([10*i for i in range(201)], np.array(train_accs[id][s]) - np.array(val_accs[id][s]))
                plt.title(id)
                plt.axvline(id[0], color='k', linestyle='--')
                plt.ylabel('Accuracy gap')

                # # reg value
                # plt.figure()
                # for s in range(len(v)):
                #     plt.plot([10*i for i in range(201)], reg_values[id][s])
                # plt.title(id)
                # plt.axvline(id[0], color='k', linestyle='--')
                # plt.ylabel('Penalty')
                plt.show()


def plot_Ex1_solution(records):

    def errors(w, w_hat):
        w = w.view(-1)
        w_hat = w_hat.view(-1)

        i_causal = torch.where(w != 0)[0].view(-1)
        i_noncausal = torch.where(w == 0)[0].view(-1)

        if len(i_causal):
            error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
            error_causal = error_causal.item()
        else:
            error_causal = 0

        if len(i_noncausal):
            error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
            error_noncausal = error_noncausal.item()
        else:
            error_noncausal = 0

        return error_causal, error_noncausal

    grouped_records = get_grouped_anneal(records)

    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in alg_names if n==algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    step = collections.defaultdict(lambda: [])
    loss = collections.defaultdict(lambda: [])
    loss_causal = collections.defaultdict(lambda: [])
    loss_noncausal = collections.defaultdict(lambda: [])

    for exp in grouped_records:

        id = (exp['Anneal_iter'],
            exp['algorithm'],
            exp['dataset'])

        # step[id].append([])
        # loss[id].append([])
        # loss_causal[id].append([])
        # loss_noncausal[id].append([])
        
        for it in range(len(exp['records'])):

            solution = exp['records'][it]['solution']
            method_solution = exp['records'][it]['model_solution']
            scramble = exp['records'][it]['scramble']

            it_loss_causal, it_loss_noncausal = errors(torch.tensor(solution), torch.tensor(scramble) @ torch.tensor(method_solution))

            step[id].append(exp['records'][it]['step'])
            loss[id].append(exp['records'][it]['loss'])
            loss_causal[id].append(it_loss_causal)
            loss_noncausal[id].append(it_loss_noncausal)

    for d in dataset_names:
        for id, v in loss.items():
            # Test accs
            plt.figure()
            plt.plot(step[id], loss_causal[id])
            plt.title(id)
            # plt.axvline(id[0], color='k', linestyle='--')
            plt.ylabel('Causal loss')

            # acc gap
            plt.figure()
            plt.plot(step[id], loss_noncausal[id])
            plt.title(id)
            # plt.axvline(id[0], color='k', linestyle='--')
            plt.ylabel('nonCausal loss')

            plt.figure()
            plt.plot(step[id], loss[id])
            plt.title(id)
            # plt.axvline(id[0], color='k', linestyle='--')
            plt.ylabel('Loss')
            plt.show()



if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)
    # plot_Ex1_solution(records)
    plot_anneal_experiment(records)
    # plot_anneal_experiment_max(records)
    plot_training_curve(records, 'ERM')
   
    # if args.latex:
    #     print("\\documentclass{article}")
    #     print("\\usepackage{booktabs}")
    #     print("\\usepackage{adjustbox}")
    #     print("\\begin{document}")
    #     print("\\section{Full DomainBed results}") 
    #     print("% Total records:", len(records))
    # else:
    #     print("Total records:", len(records))

    # SELECTION_METHODS = [
    #     model_selection.IIDAccuracySelectionMethod,
    #     model_selection.LeaveOneOutSelectionMethod,
    #     model_selection.OracleSelectionMethod,
    # ]

    # for selection_method in SELECTION_METHODS:
    #     if args.latex:
    #         print()
    #         print("\\subsection{{Model selection: {}}}".format(
    #             selection_method.name)) 
    #     print_results_tables(records, selection_method, args.latex)

    # if args.latex:
    #     print("\\end{document}")


