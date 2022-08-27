#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/11/2022
#
# Distributed under terms of the MIT license.

"""

"""
import typing as tp

import numpy as np
import pandas as pd


def parse_df_dict_of_list(df: pd.DataFrame) -> tp.Dict[str, tp.List[float]]:
    v_dict = {}
    for (_col, colval) in df.iteritems():
        v = np.array(colval.values)
        v = v[~np.isnan(v)]
        v_dict[_col] = v.tolist()
    return v_dict


def parse_df_list_of_list(df: pd.DataFrame) -> tp.List[tp.List[float]]:
    v_list = []
    for (_col, colval) in df.iteritems():
        v = np.array(colval.values)
        v = v[~np.isnan(v)]
        v_list.append(v.tolist())
    return v_list
