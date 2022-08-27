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
from scipy.stats import spearmanr

from .. import plotter, utils, widget

__all__ = ["show_medication_response_motor_fluctuation"]


def show_medication_response_motor_fluctuation(data_file):

    st.markdown("# medication response motor fluctuation")

    p_list = [40, 45, 50, 55, 60]
    x_ticks_list = list(range(0, 96, 8))
    label_list = ["4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "0", "2"]

    st.markdown("## intra-day gait speed and hauser diary")

    for pd_name in ["pd1", "pd2", "pd3", "pd4"]:
        data_dict = utils.parse_df_dict_of_list(
            pd.read_excel(data_file, f"fig5a_{pd_name}_gait")
        )

        data = np.array([data_dict[f"{p}%"] for p in p_list]).transpose()

        st.markdown("**{}**".format(pd_name))
        with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
            fig = plotter.draw_progress_with_percentile(
                v_list=data.tolist(),
                x_ticks_list=x_ticks_list,
                label_list=label_list,
                num_percentile=len(p_list),
                color="tab:red",
                color_conf="tab:blue",
                alpha_list=[0.3, 0.5],
                fig_size=(10, 4.5),
                y_range=[0.3, 1.1],
                x_range=[0, 24 * 4],
            )
            st.pyplot(fig)

        data_dict = utils.parse_df_dict_of_list(
            pd.read_excel(data_file, f"fig5a_{pd_name}_diary")
        )
        prob_on_off_sleep_list = [
            [o, f, s]
            for o, f, s in zip(data_dict["on"], data_dict["off"], data_dict["sleep"])
        ]

        step = int(len(prob_on_off_sleep_list) / len(label_list))
        diary_label_list = ["" for _ in range(len(prob_on_off_sleep_list))]
        for i, l in enumerate(label_list):
            diary_label_list[i * step] = l

        with plt.style.context(["sci_transl_med.mplstyle"]):
            fig = plotter.draw_stacked_normalized_bar(
                prob_on_off_sleep_list,
                bar_width=0.8,
                label_list=diary_label_list,
                fig_size=(5, 4),
                x_range=[-0.5, 47.5],
                y_range=[0, 1],
                color_list=["tab:red", "#C58F00", "silver"],
            )
            st.pyplot(fig)

    st.markdown("## intra-day gait speed and change in medication")

    st.markdown(f"### case1")

    for time in ["before", "after"]:
        st.markdown("**{}**".format(time))
        data_dict = utils.parse_df_dict_of_list(
            pd.read_excel(data_file, f"fig5b_{time}")
        )
        data = np.array([data_dict[f"{p}%"] for p in p_list]).transpose()
        with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
            fig = plotter.draw_progress_with_percentile(
                v_list=data.tolist(),
                x_ticks_list=x_ticks_list,
                label_list=label_list,
                num_percentile=len(p_list),
                color="tab:red",
                color_conf="tab:blue",
                alpha_list=[0.3, 0.5],
                fig_size=(10, 4.5),
                y_range=[0.5, 1.0],
                x_range=[0, 24 * 4],
            )
            st.pyplot(fig)

    st.markdown(f"### case2")

    for time in ["before", "after"]:
        st.markdown("**{}**".format(time))
        data_dict = utils.parse_df_dict_of_list(
            pd.read_excel(data_file, f"figS2_{time}")
        )
        data = np.array([data_dict[f"{p}%"] for p in p_list]).transpose()
        with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
            fig = plotter.draw_progress_with_percentile(
                v_list=data.tolist(),
                x_ticks_list=x_ticks_list,
                label_list=label_list,
                num_percentile=len(p_list),
                color="tab:red",
                color_conf="tab:blue",
                alpha_list=[0.3, 0.5],
                fig_size=(10, 4.5),
                y_range=[0.5, 1.0],
                x_range=[0, 24 * 4],
            )
            st.pyplot(fig)

    st.markdown("## functional impact motor fluctuation")

    data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig5_v_var_vs_motor_fluc")
    )
    data_list = [data_dict[str(i)] for i in range(5)]

    with plt.style.context(["sci_transl_med.mplstyle"]):
        label_list = ["normal", "slight", "mild", "moderate", "severe"]
        medianprops = dict(linestyle="-", linewidth=3, color="red")
        fig = plotter.draw_list_of_boxplots(
            data_list=data_list,
            point_color="tab:blue",
            point_alpha=0.4,
            point_size=5,
            label_list=label_list,
            fig_size=(8, 8),
            medianprops=medianprops,
        )
        st.pyplot(fig)

    v_list = sum(data_list, [])
    score_list = sum([[i] * len(data_list[i]) for i in range(len(data_list))], [])
    r, p = spearmanr(v_list, score_list)
    st.markdown("spearmanr (r, p): {}".format((r, p)))
