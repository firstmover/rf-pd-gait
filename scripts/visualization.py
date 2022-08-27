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
import pandas as pd
import streamlit as st

from server import pages


def main():

    data_file = st.cache(pd.ExcelFile, allow_output_mutation=True)("./data/data.xlsx")

    options = [
        "test_retest_reliability",
        "baseline_cross_sectional",
        "longitudinal_analysis",
        "medication_response_motor_fluctuation",
        "covid_lock_down",
        "hospitalization",
        "ppmi_longitudinal_analysis",
    ]
    sel_options = st.sidebar.selectbox("Results", options)

    if sel_options == "test_retest_reliability":
        pages.show_test_retest_reliability(data_file)
    elif sel_options == "baseline_cross_sectional":
        pages.show_baseline_cross_sectional(data_file)
    elif sel_options == "longitudinal_analysis":
        pages.show_longitudinal_analysis(data_file)
    elif sel_options == "medication_response_motor_fluctuation":
        pages.show_medication_response_motor_fluctuation(data_file)
    elif sel_options == "covid_lock_down":
        pages.show_covid_lock_down(data_file)
    elif sel_options == "hospitalization":
        pages.show_hospitalization(data_file)
    elif sel_options == "ppmi_longitudinal_analysis":
        pages.show_ppmi_longitudinal_analysis()
    else:
        raise RuntimeError


if __name__ == "__main__":
    main()
