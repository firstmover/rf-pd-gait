#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/12/2022
#
# Distributed under terms of the MIT license.

"""

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import wilcoxon

from .. import plotter, utils, widget

__all__ = ["show_covid_lock_down"]


def get_median_iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return np.median(x), q75 - q25


def show_covid_lock_down(data_file):

    st.markdown("# covid lockdown")

    fig_size = widget.text_input_tuple_float("fig size", (3, 4.5))

    for m_name in ["num_walking", "time_in_bed"]:
        st.markdown(f"## {m_name}")

        data = utils.parse_df_list_of_list(pd.read_excel(data_file, f"fig6a_{m_name}"))

        data = np.array(data).transpose().tolist()

        y_range = widget.text_input_tuple_float("y range", (0, 0), key=m_name)
        if y_range == (0, 0):
            y_range = None

        with plt.style.context(["sci_transl_med.mplstyle"]):
            fig = plotter.draw_paired(
                data,
                label_paired=["pre-lockdown", "post-lockdown"],
                y_range=y_range,
                fig_size=fig_size,
            )
            st.pyplot(fig)

        pre_data = [d[0] for d in data]
        median, iqr = get_median_iqr(pre_data)
        st.markdown("median, iqr: {:.4f}, {:.4f}".format(median, iqr))

        post_data = [d[1] for d in data]
        median, iqr = get_median_iqr(post_data)
        st.markdown("median, iqr: {:.4f}, {:.4f}".format(median, iqr))

        increase = [d[1] - d[0] for d in data]
        st.markdown("wilcoxon less: {}".format(wilcoxon(increase, alternative="less")))
        st.markdown(
            "wilcoxon greater: {}".format(wilcoxon(increase, alternative="greater"))
        )
