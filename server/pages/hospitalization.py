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

from .. import plotter, utils

__all__ = ["show_hospitalization"]


def show_hospitalization(data_file):

    st.markdown("# hospitalization")

    p_list = [30, 40, 45, 50, 55, 60, 70]
    data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig6b_hospitalization")
    )

    data = np.array([data_dict[f"{p:d}"] for p in p_list]).transpose()

    with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
        y_ticks_list = [0.5, 0.6, 0.7, 0.8]
        x_ticks_list = list(range(0, 36, 2))
        step_label_list = list(range(-14, 2, 2)) + list(range(10, 30, 2))

        fig = plotter.draw_gait_speed_progress_conf_interval_for_hospitalization(
            v_list=data,
            label_list=step_label_list,
            x_ticks_list=x_ticks_list,
            y_ticks_list=y_ticks_list,
            num_percentile=len(p_list),
            color="tab:red",
            color_conf="tab:blue",
            alpha_list=[0.15, 0.35, 0.55],
            fig_size=(10, 4),
            y_range=(0.5, 0.8),
            x_range=[0, len(data) - 1],
        )
        st.pyplot(fig)
