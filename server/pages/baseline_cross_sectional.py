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

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr, ttest_ind

from .. import plotter, utils, widget

__all__ = ["show_baseline_cross_sectional"]


def show_baseline_cross_sectional(data_file):

    st.markdown("# baseline cross sectional analysis")

    st.markdown("## pd vs hc")

    data_dict = utils.parse_df_dict_of_list(pd.read_excel(data_file, "figs3_pd_vs_hc"))

    pd_v_list = data_dict["pd"]
    hc_v_list = data_dict["hc"]

    t, p = ttest_ind(pd_v_list, hc_v_list)
    st.markdown("t, p: {:.4f}, {:.4f}".format(t, p))

    fig_size = widget.text_input_tuple_float(
        "fig size", (5, 8), "pd vs hc spd fig size"
    )
    y_range = widget.text_input_tuple_float(
        "y range", (0.40, 1.2), "pd vs hc spd y range"
    )
    seed = st.number_input("seed", value=3, min_value=0, max_value=100)

    with plt.style.context(["sci_transl_med.mplstyle"]):
        fig = plotter.draw_boxplot_with_samples(
            [pd_v_list, hc_v_list],
            label_list=["PD", "HC"],
            color_list=["tab:red", "tab:blue"],
            seed=seed,
            box_width=0.5,
            y_range=y_range,
            fig_size=fig_size,
        )
        st.pyplot(fig)

    st.markdown("## in home gait speed vs in clinic measurements")

    data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig3_gait_and_clinic")
    )
    assess_name_list = list(sorted(data_dict.keys()))

    x_name = st.selectbox("x name", assess_name_list, index=5)
    y_name = st.selectbox("y name", assess_name_list, index=1)
    x_list = data_dict[x_name]
    y_list = data_dict[y_name]

    with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
        fig = plotter.plot_scatter_plot_and_conf_interval(
            x_list=x_list,
            y_list=y_list,
            fig_size=(6, 6),
            x_range=None,
            y_range=None,
        )
        st.pyplot(fig)

    num_sample = len(x_list)
    st.markdown("n = {}".format(num_sample))

    cor, p = pearsonr(y_list, x_list)
    st.markdown("cor = {:.4f}, p = {:.4f}".format(cor, p))

    name_list = ["in_home_spd", "updrs_part3", "updrs_total", "hy_stage", "tug", "tmwt"]
    result_list = []
    corr_matrix: tp.List[tp.List[float]] = []
    for i, name in enumerate(name_list):
        x_list = data_dict[name]
        r_list = []
        corr_list: tp.List[float] = []
        for j, n in enumerate(name_list):
            y_list = data_dict[n]
            corr, p = pearsonr(x_list, y_list)
            corr_list.append(corr)
            r_list.append("{:.4f} {:.4f}".format(corr, p))
        result_list.append(r_list)
        corr_matrix.append(corr_list)
    st.table(result_list)

    with plt.style.context(["sci_transl_med.mplstyle"]):
        fig = plotter.plot_correlation_matrix_heatmap(corr_matrix)
        st.pyplot(fig)
