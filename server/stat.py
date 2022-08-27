#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/10/2022
#
# Distributed under terms of the MIT license.

"""

"""
import typing as tp

import pandas as pd
import pingouin as pg


def get_icc(
    measurements: tp.List[tp.List[float]],
    icc_type: tp.Optional[str] = None,
    nan_policy: tp.Optional[str] = None,
    min_repeat: int = 3,
) -> tp.Tuple[float, float, tp.Tuple[float, float]]:
    # first dimention of measurements represents target, for example, subject

    if icc_type is None:
        icc_type = "ICC2k"
    assert icc_type in ["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]

    if nan_policy is None:
        nan_policy = "raise"

    measurements = [m for m in measurements if len(m) >= min_repeat]
    min_n = min([len(m) for m in measurements])
    measurements = [m[:min_n] for m in measurements]

    measurement_matrix = []
    for idx_measure, measure in enumerate(measurements):
        for i, m in enumerate(measure):
            measurement_matrix.append([idx_measure, m, i])

    df = pd.DataFrame(measurement_matrix, columns=["idx_measure", "value", "idx_rator"])
    icc = pg.intraclass_corr(
        data=df,
        targets="idx_measure",
        raters="idx_rator",
        ratings="value",
        nan_policy=nan_policy,
    ).round(3)
    icc.set_index("Type")

    idx = list(icc["Type"]).index(icc_type)
    type_icc = icc["ICC"][idx]
    p_value = icc["pval"][idx]
    conf_interval_95_percent = icc["CI95%"][idx]
    return type_icc, p_value, conf_interval_95_percent
