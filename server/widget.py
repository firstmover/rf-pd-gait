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

import streamlit as st


def text_input_tuple_float(
    name: str,
    default_tuple_float: tp.Tuple[float, float],
    key: tp.Optional[str] = None,
    ran: tp.Optional[tp.Tuple[float, float]] = None,
    use_sidebar: bool = False,
):

    if ran is not None:
        for f in default_tuple_float:
            assert ran[0] <= f <= ran[1]

    if key is None:
        key = name

    st_func = st.sidebar.text_input if use_sidebar else st.text_input
    str_float = st_func(
        name,
        "({:.4f}, {:.4f})".format(default_tuple_float[0], default_tuple_float[1]),
        key=key,
    )
    str_float = str_float.replace("(", "").replace(")", "")
    float_list = [float(i) for i in str_float.split(",")]

    if ran is not None:
        for f in float_list:
            assert ran[0] <= f <= ran[1]

    return tuple(float_list)
