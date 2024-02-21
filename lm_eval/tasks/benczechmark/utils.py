# -*- coding: UTF-8 -*-
"""

:authors:     Martin Dočekal, Martin Fajčík
"""
import numpy
import evaluate
import datasets

from sklearn.metrics import f1_score, confusion_matrix
from lm_eval.api.registry import register_aggregation, register_metric
from typing import Optional


# The f1_posterior and _evaluate_statistics implementation is based on [GOUTTE-2005], and these few lines were borrowed
# and modified from Andre Anjos <anjos@idiap.ch> under Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/

def f1_posterior(tp, fp, fn, lambda_, nb_samples):
    """Simulates the F1-score posterior of a system with the provided markings

    This implementation is based on [GOUTTE-2005]_, equation 11.

    Parameters
    ----------

    tp : int
        True positive count, AKA "hit"

    fp : int
        False positive count, AKA "false alarm", or "Type I error"

    fn : int
        False Negative count, AKA "miss", or "Type II error"

    lambda_ : float
        The parameterisation of the Beta prior to consider. Use
        :math:`\lambda=1` for a flat prior.  Use :math:`\lambda=0.5` for
        Jeffrey's prior.  If unsure, use 0.5.

    nb_samples : int
        number of generated gamma distribution values


    Returns
    -------

    variates : numpy.ndarray
        An array with size ``nb_samples`` containing a realization of equation
        11.

    """

    u = numpy.random.gamma(shape=(tp + lambda_), scale=2.0, size=nb_samples)
    v = numpy.random.gamma(
        shape=(fp + fn + (2 * lambda_)), scale=1.0, size=nb_samples
    )
    return u / (u + v)


def _evaluate_statistics(variates, coverage):
    """Evaluates the left and right margins for a given M-C distribution


    Parameters
    ----------

    variates : numpy.ndarray
        A 1-D array containing the simulated variates

    coverage : float
        A number, between 0 and 1 to indicate the desired coverage.  Typically,
        this number is set to 0.95 (95% coverage).


    Returns
    -------

    stats : (float, float, float, float)
        mean, mode and credible intervals for the input simulation

    """

    left_half = (1 - coverage) / 2  # size of excluded (half) area
    sorted_variates = numpy.sort(variates)

    # n.b.: we return the equally tailed range

    # calculates position of score which would exclude the left_half (left)
    lower_index = int(round(len(variates) * left_half))

    # calculates position of score which would exclude the right_half (right)
    upper_index = int(round(len(variates) * (1 - left_half)))

    lower = sorted_variates[lower_index - 1]
    upper = sorted_variates[upper_index - 1]

    return lower, upper


def macro_f1_score(items, **kwargs):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average='macro')
    return fscore


def macro_f1_CI(items, alpha=0.95):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]

    # Calculate confusion matrix
    cm = confusion_matrix(golds, preds)

    # Get unique labels
    unique_labels = numpy.unique(golds + preds)

    # Iterate over confusion matrix to compute metrics for each class
    samples = []
    for i in range(len(unique_labels)):
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP

        # get samples from binary F1 distribution
        samples.append(f1_posterior(TP, FP, FN, 1, 10000))  # 1 = flat prior

    # convert binary f1 samples to macro f1 samples
    samples = numpy.array(samples)
    samples = numpy.mean(samples, axis=0)

    # estimate the credible interval
    lower, upper = _evaluate_statistics(samples, alpha)

    return (lower, upper)


def rouge_raw_r1_f(predictions, references):
    return rouge_raw(predictions, references, "rougeraw1_fmeasure")


def rouge_raw_r2_f(predictions, references):
    return rouge_raw(predictions, references, "rougeraw2_fmeasure")


def rouge_raw_rl_f(predictions, references):
    return rouge_raw(predictions, references, "rougerawl_fmeasure")


def rouge_raw(predictions, references, select: Optional[str] = None):
    module = evaluate.load("CZLC/rouge_raw")
    return module.compute(predictions=predictions, references=references, select=select)


def make_toy_dataset(dataset: datasets.Dataset):
    """
    Makes a toy dataset from given dataset. It means that the dataset will contain only 10 samples.

    :param dataset: Dataset to make toy from.
    """
    return dataset.select(range(10))
