#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/13/2022
#
# Distributed under terms of the MIT license.

"""

"""
import typing as tp

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm

import plotly.express as px

from ppmi.data import PPMIMDSUPDRS
from ppmi.record import MDSUPDRSSubScore

__all__ = ["show_ppmi_longitudinal_analysis"]


def _parse_score(
    pid2evt2sub_scores: tp.Dict[str, tp.Dict[str, tp.List[float]]],
    pid2evt2all_sub_scores: tp.Dict[str, tp.Dict[str, MDSUPDRSSubScore]],
    included_evt_name_list: tp.List[str],
    evt_name2time: tp.Dict[str, int],
    return_flat: bool = True,
):

    pid_list_list = []
    t_list_list = []
    total_list_list = []
    part3_list_list = []
    is_med_list_list = []

    for pid, evt2sub_score in pid2evt2sub_scores.items():

        pid_list = []
        t_list = []
        total_list = []
        part3_list = []
        is_med_list = []

        for evt, sub_score in evt2sub_score.items():
            if evt not in included_evt_name_list:
                continue
            sub_score = list(sub_score)

            # NOTE(YL 03/02):: replace None with nan
            sub_score = np.array(sub_score, dtype=np.float)

            part3_list.append(sub_score[2])
            total_list.append(np.sum(sub_score))
            pid_list.append(pid)
            t_list.append(evt_name2time[evt])
            is_med_list.append(
                pid2evt2all_sub_scores[pid][evt].sub_scores[2]["Is Taking Medication"]
            )

        pid_list = np.array(pid_list, dtype=np.int)
        t_list = np.array(t_list, dtype=np.int)
        total_list = np.array(total_list, dtype=np.float)
        part3_list = np.array(part3_list, dtype=np.float)
        is_med_list = np.array(is_med_list, dtype=np.bool)

        pid_list_list.append(pid_list)
        t_list_list.append(t_list)
        total_list_list.append(total_list)
        part3_list_list.append(part3_list)
        is_med_list_list.append(is_med_list)

    if return_flat:
        pid_list_list = np.concatenate(pid_list_list)
        t_list_list = np.concatenate(t_list_list)
        part3_list_list = np.concatenate(part3_list_list)
        total_list_list = np.concatenate(total_list_list)
        is_med_list_list = np.concatenate(is_med_list_list)

    return (
        t_list_list,
        part3_list_list,
        total_list_list,
        pid_list_list,
        is_med_list_list,
    )


def _linear_mixed_effects_model(
    y: tp.List[float], x: tp.List[float], group: tp.List[int]
):
    assert len(y) == len(x) and len(y) == len(group)

    df = pd.DataFrame({"x": x, "y": y, "group": group})
    formula = "y ~ x"
    mod_lme = MixedLM.from_formula(formula, groups=df["group"], data=df)
    mod_lme = mod_lme.fit()

    return float(mod_lme.params["x"]), float(mod_lme.pvalues["x"])


def sample_participants_and_linear_regress(
    pid_list: tp.List[float],
    t_list: tp.List[float],
    score_list: tp.List[float],
    num_participant: int,
    num_repeat: int,
) -> tp.Dict[str, tp.List[float]]:

    unique_pid_list = list(set(pid_list))

    stat_ret_list = {"slope": [], "intercept": [], "p": []}
    for _ in tqdm(range(num_repeat)):

        sampled_pid_idx_list = np.random.choice(len(unique_pid_list), num_participant)
        sampled_pid_set = set([unique_pid_list[i] for i in sampled_pid_idx_list])

        sampled_data_idx_list = [
            i for i, pid in enumerate(pid_list) if pid in sampled_pid_set
        ]
        sampled_t_list = [t_list[i] for i in sampled_data_idx_list]
        sample_score_list = [score_list[i] for i in sampled_data_idx_list]

        slope, intercept, r, p, se = stats.linregress(sampled_t_list, sample_score_list)

        stat_ret_list["slope"].append(slope)
        stat_ret_list["intercept"].append(intercept)
        stat_ret_list["p"].append(p)

    return stat_ret_list


def show_ppmi_longitudinal_analysis():

    st.markdown("## PPMI MDS-UPDRS longitudinal analysis")

    pd_subgroup = st.selectbox("pd cohort", ["all", "sporadic", "genetic"])

    ppmi_mds_updrs = st.cache(PPMIMDSUPDRS, allow_output_mutation=True)(
        pd_subgroup=pd_subgroup
    )
    pid2evt2sub_scores = ppmi_mds_updrs.pd_pid2evt2sub_scores
    pid2evt2all_sub_scores = ppmi_mds_updrs.pd_pid2evt2all_sub_scores

    evt_name2time = {
        "Baseline": 0,
        "Month 12": 1,
        "Month 24": 2,
        "Month 36": 3,
        "Month 48": 4,
        "Month 60": 5,
    }

    all_evt_name_list = list(evt_name2time.keys())
    included_evt_name_list = st.multiselect(
        "include evt names", all_evt_name_list, all_evt_name_list
    )

    t_list, part3_list, total_list, pid_list, is_med_list = _parse_score(
        pid2evt2sub_scores,
        pid2evt2all_sub_scores,
        included_evt_name_list,
        evt_name2time,
        return_flat=True,
    )

    df = pd.DataFrame(
        {
            "pid": pid_list,
            "t": t_list,
            "part3": part3_list,
            "total": total_list,
            "is_med": is_med_list,
        }
    )
    st.dataframe(df)

    score_name_list = ["part3", "total"]
    sel_score = st.selectbox("score", score_name_list, index=0)

    if sel_score == score_name_list[0]:
        score_list = part3_list
    elif sel_score == score_name_list[1]:
        score_list = total_list
    else:
        raise RuntimeError

    t_list = np.array(t_list).astype(float)
    score_list = np.array(score_list).astype(float)
    pid_list = np.array(pid_list).astype(int)

    is_not_nan = ~np.isnan(score_list)
    st.markdown(
        "num sample not nan / all = {} / {}".format(np.sum(is_not_nan), len(pid_list))
    )
    t_list = t_list[is_not_nan]
    score_list = score_list[is_not_nan]
    pid_list = pid_list[is_not_nan]

    st.markdown("### per visit data")
    unique_t_list = np.unique(t_list)
    cnt_list = [np.sum(t_list == t) for t in unique_t_list]
    mean_list = [np.mean(score_list[t_list == t]) for t in unique_t_list]
    std_list = [np.std(score_list[t_list == t]) for t in unique_t_list]
    df_stat = pd.DataFrame({"cnt": cnt_list, "mean": mean_list, "std": std_list})
    st.table(df_stat)

    fig = px.box(x=t_list, y=score_list, points="all")
    st.plotly_chart(fig)

    st.markdown("### linear mixed effect model (all data)")
    coeffi, pvalue = _linear_mixed_effects_model(y=score_list, x=t_list, group=pid_list)
    st.markdown("slope: {:.4f}".format(coeffi))
    st.markdown("pvalue: {:.4f}".format(pvalue))

    st.markdown("### linear mixed effect model (random sample simulation)")
    num_sample_list = [30, 100, 200, 300, 500]
    data_dict = {
        "slope_mean": [],
        "slope_std": [],
        "intercept_mean": [],
        "intercept_std": [],
        "p_mean": [],
        "p_std": [],
    }
    for n in num_sample_list:
        stat_ret = sample_participants_and_linear_regress(
            pid_list, t_list, score_list, n, 1000
        )
        data_dict["slope_mean"].append(np.mean(stat_ret["slope"]))
        data_dict["slope_std"].append(np.std(stat_ret["slope"]))
        data_dict["intercept_mean"].append(np.mean(stat_ret["intercept"]))
        data_dict["intercept_std"].append(np.std(stat_ret["intercept"]))
        data_dict["p_mean"].append(np.mean(stat_ret["p"]))
        data_dict["p_std"].append(np.std(stat_ret["p"]))

    df = pd.DataFrame({"num_sample": num_sample_list, **data_dict})
    st.write(df)
