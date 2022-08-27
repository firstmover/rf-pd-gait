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
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .. import plotter, stat, utils, widget

__all__ = ["show_test_retest_reliability"]


def show_test_retest_reliability(data_file):

    st.markdown("# test-retest reliability")

    st.markdown("## R w.r.t. window size")

    df = utils.parse_df_dict_of_list(pd.read_excel(data_file, "fig2a_icc"))
    icc_list = df["icc"]
    conf_interval_up_list = df["conf_interval_up"]
    conf_interval_low_list = df["conf_interval_low"]
    error_list = [
        (u - l) / 2 for u, l in zip(conf_interval_up_list, conf_interval_low_list)
    ]

    fig_size = widget.text_input_tuple_float("fig size", (10, 9), "icc fig size")
    y_range = widget.text_input_tuple_float("y range", (0.70, 1), "icc y range")

    with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
        fig = plotter.draw_curve(
            x_list=list(range(1, len(icc_list) + 1)),
            y_list=icc_list,
            y_error_list=error_list,
            y_range=y_range,
            x_range=[0, len(icc_list) + 1],
            fig_size=fig_size,
        )
        st.pyplot(fig)

    st.markdown("## ICC")

    rep_measurement_list = utils.parse_df_list_of_list(
        pd.read_excel(data_file, "fig2a_14days_measurements")
    )
    icc, pvalue, conf_interval = stat.get_icc(
        rep_measurement_list,
        icc_type="ICC1",
    )
    st.markdown(f"icc: {icc:.4f}, pvalue: {pvalue:.4f}, conf_interval: {conf_interval}")

    st.markdown("## pd and control individual results")

    pd_v_list = utils.parse_df_list_of_list(pd.read_excel(data_file, "fig2b_pd"))
    hc_v_list = utils.parse_df_list_of_list(pd.read_excel(data_file, "fig2b_hc"))

    fig_size = widget.text_input_tuple_float(
        "fig size",
        (10, 4),
        "test-retest",
    )
    with plt.style.context(["sci_transl_med.mplstyle"]):
        label_list = [str(i) for i in range(1, len(pd_v_list) + 1)]
        fig = plotter.draw_progress_boxplot(
            v_list=pd_v_list,
            label_list=label_list,
            fig_size=fig_size,
            y_range=(0.5, 1.0),
        )
        st.pyplot(fig)

        label_list = [str(i) for i in range(1, len(hc_v_list) + 1)]
        fig = plotter.draw_progress_boxplot(
            v_list=hc_v_list,
            label_list=label_list,
            fig_size=fig_size,
            y_range=(0.6, 1.1),
        )
        st.pyplot(fig)
