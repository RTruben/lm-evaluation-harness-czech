# -*- coding: UTF-8 -*-
"""
Created on 02.02.24

:author:     Martin Dočekal
"""
import evaluate
import datasets
from typing import Optional


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