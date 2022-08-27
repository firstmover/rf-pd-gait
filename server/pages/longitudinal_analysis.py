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
from scipy.stats import linregress

from .. import plotter, utils, widget

__all__ = ["show_longitudinal_analysis"]


def show_longitudinal_analysis(data_file):

    st.markdown("# longitudinal analysis")

    st.markdown("## individual results")

    fig_size = widget.text_input_tuple_float("fig size", (5, 6))

    label2range = {"pd": (0.54, 0.64), "hc": (0.76, 0.86)}
    for label in ["pd", "hc"]:

        data_dict = utils.parse_df_dict_of_list(
            pd.read_excel(data_file, f"fig4a_{label}")
        )

        v = data_dict["v"]
        t = data_dict["t"]

        slope, _intercept, _r_value, p_value, std_err = linregress(t, v)

        slope = slope * 12
        std_err = std_err * 12
        st.markdown(
            "{:.4f} ([{:.4f}, {:.4f}] 95% CI, p = {:.4f})".format(
                slope, slope - 1.96 * std_err, slope + 1.96 * std_err, p_value
            )
        )

        with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
            fig = plotter.draw_progression_with_scatter_plot_and_conf_interval(
                x_list=t,
                v_list=v,
                conf_level=0.95,
                color="tab:red",
                color_ci="tab:blue",
                alpha_ci=0.3,
                y_step_size=0.02,
                fig_size=fig_size,
                y_range=label2range[label],
                x_range=[0, 12],
            )
            st.pyplot(fig)

    st.markdown("## cohort results")

    pd_ind_data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig4b_pd_individuals")
    )
    hc_ind_data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig4b_hc_individuals")
    )
    pd_coh_data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig4b_pd_cohort")
    )
    hc_coh_data_dict = utils.parse_df_dict_of_list(
        pd.read_excel(data_file, "fig4b_hc_cohort")
    )

    with plt.style.context(["sci_transl_med.mplstyle", "grid.mplstyle"]):
        fig = plotter.draw_pd_hc_cohort_progress(
            pd_x_range_list=list(
                zip(pd_ind_data_dict["pd_x_min"], pd_ind_data_dict["pd_x_max"])
            ),
            pd_intercept_slope_list=list(
                zip(pd_ind_data_dict["pd_intercept"], pd_ind_data_dict["pd_slope"])
            ),
            pd_cohort_x=pd_coh_data_dict["pd_cohort_x"],
            pd_cohort_pred=pd_coh_data_dict["pd_cohort_pred"],
            pd_conf_low=pd_coh_data_dict["pd_conf_low"],
            pd_conf_high=pd_coh_data_dict["pd_conf_high"],
            pd_color="red",
            pd_alpha_color="tab:red",
            hc_x_range_list=list(
                zip(hc_ind_data_dict["hc_x_min"], hc_ind_data_dict["hc_x_max"])
            ),
            hc_intercept_slope_list=list(
                zip(hc_ind_data_dict["hc_intercept"], hc_ind_data_dict["hc_slope"])
            ),
            hc_cohort_x=hc_coh_data_dict["hc_cohort_x"],
            hc_cohort_pred=hc_coh_data_dict["hc_cohort_pred"],
            hc_conf_low=hc_coh_data_dict["hc_conf_low"],
            hc_conf_high=hc_coh_data_dict["hc_conf_high"],
            hc_color="blue",
            hc_alpha_color="tab:blue",
            x_range=[0, 12],
            y_range=[-0.05, 0.02],
            alpha=0.4,
            alpha_conf_interval=0.5,
            cohort_line_width=3.0,
            fig_size=(10, 8),
        )
        st.pyplot(fig)
