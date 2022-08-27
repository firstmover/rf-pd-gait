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
import functools
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def draw_curve(
    x_list: tp.List[float],
    y_list: tp.List[float],
    y_error_list: tp.List[float],
    color: str = "red",
    color_conf_interval: str = "tab:blue",
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    y_range: tp.Optional[tp.Tuple[float, float]] = None,
    x_range: tp.Optional[tp.Tuple[float, float]] = None,
):

    assert len(x_list) == len(y_list)

    if not isinstance(x_list, np.ndarray):
        x_list = np.array(x_list)

    if not isinstance(y_list, np.ndarray):
        y_list = np.array(y_list)

    if not isinstance(y_error_list, np.ndarray):
        y_error_list = np.array(y_error_list)

    if fig_size is None:
        fig_size = (10, 10)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(x_list, y_list, color=color, ls="--", marker="o", markersize=4)

    ax.fill_between(
        x_list,
        y_list - y_error_list,
        y_list + y_error_list,
        color=color_conf_interval,
        alpha=0.2,
    )

    if x_range is not None:
        ax.set_xlim(*x_range)

    if y_range is not None:
        ax.set_ylim(*y_range)

    return fig


def draw_progress_boxplot(
    v_list: tp.List[tp.List[float]],
    label_list: tp.List[str],
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
    gridspec_kw: tp.Optional[tp.Dict[str, float]] = None,
):

    if y_range is None:
        y_range = [0.2, 1.5]

    if fig_size is None:
        fig_size = (8, 6)

    fig, ax = plt.subplots(figsize=fig_size, gridspec_kw=gridspec_kw)

    medianprops = dict(linestyle="-", linewidth=3, color="red")
    flierprops = {"color": "gray", "marker": "x"}
    bplot = ax.boxplot(
        v_list,
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        labels=list(range(len(v_list))),  # will be used to label x-ticks
        manage_ticks=True,
        showfliers=True,
        medianprops=medianprops,
        flierprops=flierprops,
    )

    for p in bplot["boxes"]:
        p.set_facecolor("white")

    ax.yaxis.grid(True, linestyle="--")

    ax.set_xticklabels(label_list)

    ax.set_ylim(*y_range)

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    return fig


def draw_boxplot_with_samples(
    x_list: tp.List[tp.List[float]],
    label_list: tp.Optional[tp.List[float]] = None,
    color_list: tp.List[str] = None,
    seed: int = 0,
    box_width: float = 0.5,
    y_range: tp.Optional[tp.List[float]] = None,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
):

    if fig_size is None:
        fig_size = (8, 6)

    if y_range is None:
        y_range = [0.5, 1.2]

    if color_list is None:
        color_list = ["tab:red"] * len(x_list)
    assert len(color_list) == len(x_list)

    np.random.seed(seed)

    fig, ax = plt.subplots(figsize=fig_size)

    medianprops = dict(linestyle="-", linewidth=2, color="red")
    flierprops = {"color": "gray", "marker": "x"}
    ax.boxplot(
        x_list,
        widths=box_width,
        labels=list(range(len(x_list))),  # will be used to label x-ticks
        vert=True,  # vertical box alignment
        patch_artist=False,  # fill with color
        manage_ticks=True,
        showfliers=True,
        medianprops=medianprops,
        flierprops=flierprops,
    )

    for x, y, c in zip(x_list, range(len(x_list)), color_list):
        _y = np.random.normal(y + 1, 0.04, size=len(x))
        ax.plot(_y, x, ".", color=c, alpha=0.4)

    ax.yaxis.grid(True, linestyle="--")

    if label_list is not None:
        ax.set_xticklabels(label_list)

    ax.set_ylim(*y_range)

    if title is not None:
        ax.set_title(title)

    return fig


def draw_paired(
    x_paired_list: tp.List[tp.List],
    label_paired: tp.List[str],
    xlabel: tp.Optional[str] = None,
    ylabel: tp.Optional[str] = None,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
    plot_points: bool = True,
    point_color: str = "tab:blue",
    point_alpha: float = 0.4,
    point_size: float = 6,
):

    if fig_size is None:
        fig_size = (8, 6)

    fig, ax = plt.subplots(figsize=fig_size)

    medianprops = dict(linestyle="-", linewidth=3, color="red")
    flierprops = dict(marker="x", markersize=3)
    styled_boxplot = functools.partial(
        ax.boxplot,
        vert=True,
        widths=0.5,
        patch_artist=False,
        manage_ticks=True,
        showfliers=True,
        flierprops=flierprops,
        medianprops=medianprops,
    )

    before_data = [d[0] for d in x_paired_list]
    styled_boxplot(before_data, positions=[0])

    after_data = [d[1] for d in x_paired_list]
    styled_boxplot(after_data, positions=[1])

    if plot_points:
        _y = np.random.normal(0, 0.06, size=len(before_data))
        ax.plot(
            _y,
            before_data,
            ".",
            color=point_color,
            alpha=point_alpha,
            markersize=point_size,
        )
        _y = np.random.normal(1, 0.06, size=len(after_data))
        ax.plot(
            _y,
            after_data,
            ".",
            color=point_color,
            alpha=point_alpha,
            markersize=point_size,
        )

    ax.yaxis.grid(True)
    #  ax.xaxis.grid(True)

    if label_paired is not None:
        assert len(label_paired) == 2
        ax.set_xticklabels(label_paired)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    if y_range is not None:
        ax.set_ylim(*y_range)

    return fig


def draw_list_of_boxplots(
    data_list: tp.List[tp.List[float]],
    plot_points: bool = True,
    point_color: str = "tab:blue",
    point_alpha: float = 0.4,
    point_size: float = 10,
    label_list: tp.Optional[tp.List[str]] = None,
    medianprops: tp.Optional[tp.Dict] = None,
    flierprops: tp.Optional[tp.Dict] = None,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
):

    if fig_size is None:
        fig_size = (8, 6)

    if medianprops is None:
        medianprops = dict(linestyle="-", linewidth=3, color="red")

    if flierprops is None:
        flierprops = dict(marker="x", markersize=3)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.boxplot(
        data_list,
        vert=True,  # vertical box alignment
        #  patch_artist=True,  # fill with color
        patch_artist=False,
        manage_ticks=True,
        showfliers=True,
        medianprops=medianprops,
        flierprops=dict(marker="x", markersize=3),
    )

    if plot_points:
        for data, y in zip(data_list, range(len(data_list))):
            _y = np.random.normal(y + 1, 0.04, size=len(data))
            ax.plot(
                _y,
                data,
                ".",
                color=point_color,
                alpha=point_alpha,
                markersize=point_size,
            )

    ax.yaxis.grid(True)
    #  ax.xaxis.grid(True)

    if label_list is not None:
        ax.set_xticklabels(label_list)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    if y_range is not None:
        ax.set_ylim(*y_range)

    if title is not None:
        ax.set_title(title)

    return fig


def plot_ci_manual(t, s_err, n, x, x2, y2, ax, color_ci, alpha_ci):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \;
    \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    ax.fill_between(x2, y2 + ci, y2 - ci, color=color_ci, alpha=alpha_ci)

    return ax


def _add_conf_interval(
    x,
    y,
    a,
    conf_level: float = 0.975,
    color_ci: str = "tab:blue",
    alpha_ci: float = 0.5,
):

    equation = np.polyval

    p, cov = np.polyfit(
        x, y, 1, cov=True
    )  # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(
        p, x
    )  # model using the fit parameters; NOTE: parameters here are coefficients

    # Statistics
    n = y.size  # number of observations
    m = p.size  # number of parameters
    dof = n - m  # degrees of freedom
    t = stats.t.ppf(conf_level, n - m)  # used for CI and PI bands

    # Estimates of Error in Data/Model
    resid = y - y_model
    s_err = np.sqrt(np.sum(resid**2) / dof)  # standard deviation of the error

    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = equation(p, x2)

    plot_ci_manual(t, s_err, n, x, x2, y2, ax=a, color_ci=color_ci, alpha_ci=alpha_ci)


def plot_scatter_plot_and_conf_interval(
    x_list: tp.List[float],
    y_list: tp.List[float],
    annotations: tp.Optional[tp.List[str]] = None,
    anno_size: tp.Optional[float] = None,
    conf_level: float = 0.95,
    color_ci: str = "tab:blue",
    alpha_ci: float = 0.3,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
    y_range: tp.Optional[tp.Tuple[float, float]] = None,
    x_range: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
):

    assert len(x_list) == len(y_list)

    if fig_size is None:
        fig_size = (12, 12)

    fig, ax = plt.subplots(figsize=fig_size)

    coef = np.polyfit(x_list, y_list, 1)
    poly1d_fn = np.poly1d(coef)
    _x = np.linspace(np.min(x_list), np.max(x_list), 100)
    ax.plot(_x, poly1d_fn(_x), "--")

    _add_conf_interval(
        np.array(x_list),
        np.array(y_list),
        ax,
        conf_level=conf_level,
        color_ci=color_ci,
        alpha_ci=alpha_ci,
    )

    ax.scatter(x_list, y_list, color="r", s=10)

    if annotations is not None:
        if anno_size is None:
            anno_size = 12
        assert len(annotations) == len(x_list)
        for anno, _x, _y in zip(annotations, x_list, y_list):
            ax.annotate(anno, (_x, _y), fontsize=anno_size)

    if y_range is not None:
        ax.set_ylim(*y_range)

    if x_range is not None:
        ax.set_xlim(*x_range)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    if title is not None:
        ax.set_title(title)

    return fig


def plot_correlation_matrix_heatmap(
    data: tp.List[tp.List[float]],
    fig_size: tp.Optional[tp.Tuple[int, int]] = None,
):

    if fig_size is None:
        fig_size = (12, 10)

    fig, ax = plt.subplots(figsize=fig_size)

    sns.heatmap(
        data,
        #  mask=mask,
        #  annot=True,
        #  annot_kws={"size": 20},
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        cbar_kws={"shrink": 0.95},
        linewidths=2,
        linecolor="black",
        square=True,
    )

    return fig


def draw_progress_with_percentile(
    v_list: tp.List[tp.List[float]],
    num_percentile: int,
    color: str,
    color_conf: str,
    alpha_list: tp.List[float],
    x_ticks_list: tp.Optional[tp.List[float]] = None,
    y_ticks_list: tp.Optional[tp.List[float]] = None,
    label_list: tp.Optional[tp.List[str]] = None,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
    x_range: tp.Optional[tp.List[float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
):

    if y_range is None:
        y_range = [0.2, 1.5]

    if fig_size is None:
        fig_size = (8, 6)

    assert all([len(v) == num_percentile for v in v_list if len(v) > 0])
    assert len(alpha_list) == int(num_percentile / 2)

    fig, ax = plt.subplots(figsize=fig_size)

    x_list = []
    idx_med = int(num_percentile / 2)
    med_v_list = []
    for i, v in enumerate(v_list):
        if len(v) > 0:
            x_list.append(i)
            med_v_list.append(v[idx_med])

    for i in range(int(num_percentile / 2)):

        up_list = [v[-(i + 1)] for v in v_list if len(v) > 0]
        low_list = [v[i] for v in v_list if len(v) > 0]
        alpha = alpha_list[i]

        ax.fill_between(
            x_list,
            up_list,
            low_list,
            color=color_conf,
            alpha=alpha,
        )

    #  ax.plot(x_list, med_v_list, "--", color)
    ax.plot(x_list, med_v_list, "--", color=color, linewidth=2)

    ax.yaxis.grid(True)
    #  ax.xaxis.grid(True)

    if label_list is not None:

        if x_ticks_list is not None:
            assert len(x_ticks_list) == len(label_list)
        else:
            x_ticks_list = list(range(len(label_list)))

        ax.xaxis.set_ticks(x_ticks_list)
        ax.set_xticklabels(label_list)

    else:
        ax.xaxis.set_ticks(x_ticks_list)

    if y_ticks_list is not None:
        ax.yaxis.set_ticks(y_ticks_list)

    if y_range is not None:
        ax.set_ylim(*y_range)

    if x_range is not None:
        ax.set_xlim(*x_range)

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    return fig


def draw_progression_with_scatter_plot_and_conf_interval(
    v_list: tp.List[float],
    x_list: tp.List[float],
    conf_level: float,
    color: str,
    color_ci: str,
    alpha_ci: float,
    y_step_size: float = 0.05,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
    x_range: tp.Optional[tp.List[float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
    gridspec_kw: tp.Optional[tp.Dict[str, float]] = None,
):

    if y_range is None:
        y_range = [0.2, 1.5]

    if fig_size is None:
        fig_size = (8, 6)

    fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(x_list, v_list, color=color, ls="--", marker="o", markersize=3)

    # conf interval and linear regression

    coef = np.polyfit(x_list, v_list, 1)
    poly1d_fn = np.poly1d(coef)
    _x = np.linspace(np.min(x_list), np.max(x_list), 100)
    ax.plot(_x, poly1d_fn(_x), "--", color=color_ci, linewidth=2)

    _add_conf_interval(
        np.array(x_list),
        np.array(v_list),
        ax,
        conf_level=conf_level,
        color_ci=color_ci,
        alpha_ci=alpha_ci,
    )

    ax.set_ylim(*y_range)
    y_ticks_list = np.arange(y_range[0], y_range[1], y_step_size).tolist()
    if y_range[1] not in y_ticks_list:
        y_ticks_list.append(y_range[1])
    ax.yaxis.set_ticks(y_ticks_list)

    if x_range is not None:
        ax.set_xlim(*x_range)

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    return fig


def draw_pd_hc_cohort_progress(
    pd_x_range_list: tp.List[tp.Tuple[float, float]],
    pd_intercept_slope_list: tp.List[tp.List[float]],
    pd_cohort_x: tp.List[float],
    pd_cohort_pred: tp.List[float],
    pd_conf_low: tp.List[float],
    pd_conf_high: tp.List[float],
    pd_color: str,
    pd_alpha_color: str,
    hc_x_range_list: tp.List[tp.Tuple[float, float]],
    hc_intercept_slope_list: tp.List[tp.List[float]],
    hc_cohort_x: tp.List[float],
    hc_cohort_pred: tp.List[float],
    hc_conf_low: tp.List[float],
    hc_conf_high: tp.List[float],
    hc_color: str,
    hc_alpha_color: str,
    alpha: float = 0.3,
    alpha_conf_interval: float = 0.1,
    cohort_line_width: float = 3,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
    x_range: tp.Optional[tp.List[float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
    tick_size: int = 18,
    label_size: int = 20,
    title_size: int = 20,
):

    if y_range is None:
        y_range = [-0.1, 0.05]

    if x_range is None:
        x_range = [0, 12]

    if fig_size is None:
        fig_size = (8, 6)

    fig, ax = plt.subplots(figsize=fig_size)

    def _plot_cohort(
        x_range_list,
        intercept_slope_list,
        color,
        alpha_color,
        cohort_x,
        cohort_pred,
        conf_low,
        conf_high,
    ):
        for x_range, i_s in zip(x_range_list, intercept_slope_list):
            _x = np.linspace(x_range[0], x_range[1], 100)
            ax.plot(_x, i_s[1] * _x + i_s[0], "--", color=alpha_color, alpha=alpha)
        ax.plot(cohort_x, cohort_pred, "--", color=color, linewidth=cohort_line_width)
        ax.fill_between(
            cohort_x, conf_low, conf_high, color=alpha_color, alpha=alpha_conf_interval
        )

    _x = np.linspace(x_range[0], x_range[1], 100)
    ax.plot(_x, np.zeros_like(_x), "--", color="gray", linewidth=2)

    _plot_cohort(
        pd_x_range_list,
        pd_intercept_slope_list,
        pd_color,
        pd_alpha_color,
        pd_cohort_x,
        pd_cohort_pred,
        pd_conf_low,
        pd_conf_high,
    )

    _plot_cohort(
        hc_x_range_list,
        hc_intercept_slope_list,
        hc_color,
        hc_alpha_color,
        hc_cohort_x,
        hc_cohort_pred,
        hc_conf_low,
        hc_conf_high,
    )

    ax.set_ylim(*y_range)
    ax.set_xlim(*x_range)

    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    if title is not None:
        ax.set_title(title, fontsize=title_size)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)

    return fig


def draw_gait_speed_progress_conf_interval_for_hospitalization(
    v_list: tp.List[tp.List[float]],
    num_percentile: int,
    color: str,
    color_conf: str,
    alpha_list: tp.List[float],
    x_ticks_list: tp.Optional[tp.List[float]] = None,
    y_ticks_list: tp.Optional[tp.List[float]] = None,
    label_list: tp.Optional[tp.List[str]] = None,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
    y_range: tp.Optional[tp.List[float]] = None,
    x_range: tp.Optional[tp.List[float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
):

    if y_range is None:
        y_range = [0.2, 1.5]

    if fig_size is None:
        fig_size = (8, 6)

    len(v_list)
    assert all([len(v) == num_percentile for v in v_list if len(v) > 0])
    assert len(alpha_list) == int(num_percentile / 2)

    fig, ax = plt.subplots(figsize=fig_size)

    x_list = []
    idx_med = int(num_percentile / 2)
    med_v_list = []
    for i, v in enumerate(v_list):
        if len(v) > 0:
            x_list.append(i)
            med_v_list.append(v[idx_med])

    idx_hosp_day = np.argmin(med_v_list)

    for i in range(int(num_percentile / 2)):

        up_list = [v[-(i + 1)] for v in v_list if len(v) > 0]
        low_list = [v[i] for v in v_list if len(v) > 0]
        alpha = alpha_list[i]

        # before hospitalization
        ax.fill_between(
            x_list[: idx_hosp_day + 1],
            up_list[: idx_hosp_day + 1],
            low_list[: idx_hosp_day + 1],
            color=color_conf,
            alpha=alpha,
        )

        # after hospitalization
        ax.fill_between(
            np.array(x_list[idx_hosp_day + 1 :]) + 3,
            up_list[idx_hosp_day + 1 :],
            low_list[idx_hosp_day + 1 :],
            color=color_conf,
            alpha=alpha,
        )

    # before hospitalization
    ax.plot(
        x_list[: idx_hosp_day + 1],
        med_v_list[: idx_hosp_day + 1],
        "--",
        color=color,
        linewidth=2,
    )

    # after hospitalization
    ax.plot(
        np.array(x_list[idx_hosp_day + 1 :]) + 3,
        med_v_list[idx_hosp_day + 1 :],
        "--",
        color=color,
        linewidth=2,
    )

    ax.yaxis.grid(True)
    #  ax.xaxis.grid(True)

    if label_list is not None:

        if x_ticks_list is not None:
            assert len(x_ticks_list) == len(label_list)
        else:
            x_ticks_list = list(range(len(label_list)))

        ax.xaxis.set_ticks(x_ticks_list)
        ax.set_xticklabels(label_list)

    else:
        ax.xaxis.set_ticks(x_ticks_list)

    if y_ticks_list is not None:
        ax.yaxis.set_ticks(y_ticks_list)

    if y_range is not None:
        ax.set_ylim(*y_range)

    if x_range is not None:
        ax.set_xlim(*x_range)

    if title is not None:
        ax.set_title(title)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    return fig


def draw_stacked_normalized_bar(
    sample_list: tp.List[tp.List[float]],
    color_list: tp.List[str],
    bar_width: float = 1,
    label_list: tp.Optional[tp.List[str]] = None,
    fig_size: tp.Optional[tp.Tuple[float, float]] = None,
    ylabel: tp.Optional[str] = None,
    xlabel: tp.Optional[str] = None,
    y_range: tp.Optional[tp.Tuple[float, float]] = None,
    x_range: tp.Optional[tp.Tuple[float, float]] = None,
    title: tp.Optional[str] = None,
):

    num_class = len(sample_list[0])
    num_bar = len(sample_list)
    assert all([len(s) == num_class for s in sample_list])
    assert len(color_list) == num_class

    if label_list is not None:
        assert len(label_list) == num_bar

    if fig_size is None:
        fig_size = (12, 12)

    if not isinstance(sample_list, np.ndarray):
        sample_list = np.array(sample_list)

    fig, ax = plt.subplots(figsize=fig_size)

    ind = list(range(num_bar))
    for i in range(num_class):
        color = color_list[i]
        if i == 0:
            ax.bar(ind, sample_list[:, 0], bar_width, color=color)
        else:
            bottom = np.sum(sample_list[:, :i], axis=1)
            ax.bar(ind, sample_list[:, i], bar_width, bottom=bottom, color=color)

    if label_list is not None:
        ax.set_xticks(ind)
        ax.set_xticklabels(label_list)

    if y_range is not None:
        ax.set_ylim(*y_range)

    if x_range is not None:
        ax.set_xlim(*x_range)

    if ylabel is not None:
        ax.set(ylabel=ylabel)

    if xlabel is not None:
        ax.set(xlabel=xlabel)

    if title is not None:
        ax.set_title(title)

    return fig
