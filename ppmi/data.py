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
import copy
import typing as tp
from collections import defaultdict
from os import path as osp

import numpy as np
import pandas as pd

from .record import MDSUPDRSSubScore


class PPMIMDSUPDRS:
    cohort_name_list = ["HC", "PD"]

    def __init__(
        self,
        data_root: tp.Optional[str] = None,
        zero_imput_part4: bool = True,
        pd_subgroup: tp.Optional[str] = None,
    ):

        if data_root is None:
            proj_root = osp.dirname(osp.dirname(__file__))
            data_root = osp.join(proj_root, "data", "ppmi")

        self.data_root = data_root

        self._pid2evt2all_sub_scores, self.pid2evt2sub_scores = read_scores(
            str(self.data_root)
        )

        if zero_imput_part4:
            for p, e2scores in self.pid2evt2sub_scores.items():
                for e, scores in e2scores.items():
                    if scores[3] is None:
                        scores[3] = 0
            for p, e2scores in self._pid2evt2all_sub_scores.items():
                for e, scores in e2scores.items():
                    part4_sub_scores = scores.sub_scores[3]
                    for k, v in part4_sub_scores.items():
                        if v is None or np.isnan(v):
                            part4_sub_scores[k] = 0

        self.cohort2pid = self.read_patient_status()
        _sporadic_genetic_cohort2pid_list = (
            self.read_patient_status_consensus_committe_file()
        )

        if pd_subgroup is None or pd_subgroup == "all":
            pass
        elif pd_subgroup == "sporadic":
            self.cohort2pid["PD"] = _sporadic_genetic_cohort2pid_list["pd_sporadic"]
        elif pd_subgroup == "genetic":
            self.cohort2pid["PD"] = _sporadic_genetic_cohort2pid_list["pd_genetic"]

    @property
    def pd_pid2evt2sub_scores(self):
        pid2evt2sub_scores = {}
        for pid in self.cohort2pid["PD"]:
            if pid in self.pid2evt2sub_scores.keys():
                pid2evt2sub_scores[pid] = copy.deepcopy(self.pid2evt2sub_scores[pid])
        return pid2evt2sub_scores

    @property
    def pd_pid2evt2all_sub_scores(self):
        pid2evt2all_sub_scores = {}
        for pid in self.cohort2pid["PD"]:
            if pid in self.pid2evt2sub_scores.keys():
                pid2evt2all_sub_scores[pid] = copy.deepcopy(
                    self._pid2evt2all_sub_scores[pid]
                )
        return pid2evt2all_sub_scores

    @property
    def hc_pid2evt2sub_scores(self):
        pid2evt2sub_scores = {}
        for pid in self.cohort2pid["HC"]:
            if pid in self.pid2evt2sub_scores.keys():
                pid2evt2sub_scores[pid] = copy.deepcopy(self.pid2evt2sub_scores[pid])
        return pid2evt2sub_scores

    def read_patient_status(self) -> tp.Dict[str, tp.List[str]]:

        file_path = osp.join(self.data_root, "Participant_Status.csv")
        df = pd.read_csv(file_path)

        patient_id_list = df["PATNO"].tolist()
        cohort_list = df["COHORT"].tolist()

        cohort_label2cohort_name = {1: "PD", 2: "HC"}

        cohort_name2pid_list: tp.Dict[str, tp.List[str]] = {"PD": [], "HC": []}
        for pid, cohort in zip(patient_id_list, cohort_list):
            if int(cohort) not in cohort_label2cohort_name.keys():
                continue
            cohort_name2pid_list[cohort_label2cohort_name[int(cohort)]].append(str(pid))

        return cohort_name2pid_list

    def read_patient_status_consensus_committe_file(self) -> tp.Dict[str, tp.List[str]]:
        file_path = osp.join(
            self.data_root, "Consensus_Committee_Analytic_Datasets_28OCT21.xlsx"
        )
        df = pd.read_excel(file_path, sheet_name="PD")

        sporadic_pd_list = [
            str(i) for i in df.loc[df.Subgroup == "Sporadic"].PATNO.tolist()
        ]
        genetic_pd_list = [
            str(i) for i in df.loc[df.Subgroup == "Genetic"].PATNO.tolist()
        ]
        return {"pd_sporadic": sporadic_pd_list, "pd_genetic": genetic_pd_list}


def read_scores(
    data_root: str,
) -> tp.Tuple[
    tp.Dict[str, tp.Dict[str, MDSUPDRSSubScore]],
    tp.Dict[str, tp.Dict[str, tp.List[float]]],
]:

    evt_name2visit_name = {
        "SC": "Screening",
        "BL": "Baseline",
        "V01": "Month 3",
        "V02": "Month 6",
        "V03": "Month 9",
        "V04": "Month 12",
        "V05": "Month 18",
        "V06": "Month 24",
        "V07": "Month 30",
        "V08": "Month 36",
        "V09": "Month 42",
        "V10": "Month 48",
        "V11": "Month 54",
        "V12": "Month 60",
        "V13": "Month 72",
        "V14": "Month 84",
        "V15": "Month 96",
        "V16": "Month 108",
        "V17": "Month 120",
        "V18": "Month 132",
        "V19": "Month 144",
        "ST": "Symptomatic Therapy",
        "PW": "Premature Withdrawal",
        "U01": "Unscheduled Visit 01",
        "U02": "Unscheduled Visit 02",
        "U03": "Unscheduled Visit 03",
        "U04": "Unscheduled Visit 04",
    }

    # NOTE(YL 11/27):: let's make our life easier
    #  include_evt_entry_list = ["BL"] + ["V{:02d}".format(i) for i in range(1, 20)]
    #  include_evt_entry_list = ["SC", "BL"] + ["V04"]
    include_evt_entry_list = ["BL"] + ["V04", "V06", "V08", "V10", "V12"]

    file_name_list = [
        "MDS-UPDRS_Part_I.csv",
        "MDS-UPDRS_Part_I_Patient_Questionnaire.csv",
        "MDS_UPDRS_Part_II__Patient_Questionnaire.csv",
        "MDS_UPDRS_Part_III.csv",
        "MDS-UPDRS_Part_IV__Motor_Complications.csv",
    ]

    # parse all sub questions scores
    entry_name2question_list = [
        {
            "NP1COG": "Cognitive Impairment",
            "NP1HALL": "Hallucinations and Psychosis",
            "NP1DPRS": "Depressed Mood",
            "NP1ANXS": "Anxious Mood",
            "NP1APAT": "Apathy",
            "NP1DDS": "Dopamine Dysregulation",
            "NP1RTOT": "MDS-UPDRS Part I (Rater Completed) Total Score",
        },
        {
            "NP1SLPN": "Sleep Problems",
            "NP1SLPD": "Daytime Sleepiness",
            "NP1PAIN": "Pain and Other Sensations",
            "NP1URIN": "Urinary Problems",
            "NP1CNST": "Constipation Problems",
            "NP1LTHD": "Light Headedness on Standing",
            "NP1FATG": "Fatigue",
            "NP1PTOT": "MDS-UPDRS Part I (Patient Questionnaire) Total Score",
        },
        {
            "NP2SPCH": "Speech",
            "NP2SALV": "Salive and Drooling",
            "NP2SWAL": "Chewing and Swallowing",
            "NP2EAT": "Eating Tasks",
            "NP2DRES": "Dressing",
            "NP2HYGN": "Hygiene",
            "NP2HWRT": "Handwriting",
            "NP2HOBB": "Doing Hobbies and Other Activities",
            "NP2TURN": "Turning in Bed",
            "NP2TRMR": "Tremor",
            "NP2RISE": "Getting Out of Bed, a Car, or a Deep Chair",
            "NP2WALK": "Walking and Balance",
            "NP2FREZ": "Freezing",
            "NP2PTOT": "MDS-UPDRS Part II Total Score",
        },
        {
            "NP3SPCH": "Speech",
            "NP3FACXP": "Facial Expression",
            "NP3RIGN": "Rigidity - Neck",
            "NP3RIGRU": "Rigidity - RUE",
            "NP3RIGLU": "Rigidity - LUE",
            "NP3RIGRL": "Rigiditiy - RLE",
            "NP3RIGLL": "Rigidity - LLE",
            "NP3FTAPR": "Finger Tapping - Right",
            "NP3FTAPL": "Finger Tapping - Left",
            "NP3HMOVR": "Hand Movements - Right",
            "NP3HMOVL": "Hand Movements - Left",
            "NP3PRSPR": "Hand Pronate-Suprinate - Right",
            "NP3PRSPL": "Hand Pronate-Suprinate - Left",
            "NP3TTAPR": "Toe Tapping - Right",
            "NP3TTAPL": "Toe Tapping - Left",
            "NP3LGAGR": "Leg Agility - Right",
            "NP3LGAGL": "Leg Agility - Left",
            "NP3RISNG": "Arising from Chair",
            "NP3GAIT": "Gait",
            "NP3FRZGT": "Freezing of Gait",
            "NP3PSTBL": "Postural Stability",
            "NP3POSTR": "Posture",
            "NP3BRADY": "Body Bradykinesia",
            "NP3PTRMR": "Postural Hand Tremor - Right",
            "NP3PTRML": "Postural Hand Tremor - Left",
            "NP3KTRMR": "Kinetic Hand Tremor - Right",
            "NP3KTRML": "Kinetic Hand Tremor - Left",
            "NP3RTARU": "Rest Tremor Amplitude - RUE",
            "NP3RTALU": "Rest Tremor Amplitude - LUE",
            "NP3RTARL": "Rest Tremor Amplitude - RLE",
            "NP3RTALL": "Rest Tremor Amplitude - LLE",
            "NP3RTALJ": "Rest Tremor Amplitude - Lip/Jaw",
            "NP3RTCON": "Constancy of Rest Tremor",
            "NP3TOT": "MDS-UPDRS Part III Total Score",
            "PDTRTMNT": "Is Taking Medication",
        },
        {
            "NP4WDYSK": "Time w/ Dyskinesias",
            "NP4WDYSKNUM": "Total Hours Awake",
            "NP4WDYSKDEN": "Total Hours w/ Dyskinesia",
            "NP4WDYSKPCT": "% Dyskinesia",
            "NP4DYSKI": "Function Impact of Dyskinesias",
            "NP4OFF": "Time spent in the OFF State",
            "NP4OFFNUM": "Total Hours Awake",
            "NP4OFFDEN": "Total Hours OFF",
            "NP4OFFPCT": "% OFF",
            "NP4FLCTI": "Functional Impact of Fluctuations",
            "NP4FLCTX": "Complexity of Motor Fluctuations",
            "NP4DYSTN": "Painful OFF-state dystonia",
            "NP4DYSTNNUM": "Total Hours OFF",
            "NP4DYSTNDEN": "Total Hours OFF w/ Dystonia",
            "NP4DYSTNPCT": "% OFF Dystonia",
            "NP4TOT": "MDS-UPDRS Part IV Total Score",
        },
    ]

    pid2visit2question_sub_scores = defaultdict(
        lambda: defaultdict(lambda: MDSUPDRSSubScore(None))
    )
    pid2visit2part_sub_score = defaultdict(
        lambda: defaultdict(lambda: [None, None, None, None, None])
    )
    for i, (fname, ename2ques) in enumerate(
        zip(file_name_list, entry_name2question_list)
    ):
        for ename, ques in ename2ques.items():

            if i == 3:
                file_pid2evt2score = _read_part3_score(
                    data_root, fname, ename, include_evt_entry_list
                )
            else:
                file_pid2evt2score = _read_part_score(
                    data_root, fname, ename, include_evt_entry_list
                )

            for p, evt2score in file_pid2evt2score.items():
                for e, s in evt2score.items():
                    if str(e) not in evt_name2visit_name.keys():
                        continue
                    vname = evt_name2visit_name[str(e)]

                    if s == "UR":
                        s = None
                    else:
                        s = float(s)

                    if ename not in [
                        "NP1RTOT",
                        "NP1PTOT",
                        "NP2PTOT",
                        "NP3TOT",
                        "NP4TOT",
                    ]:
                        scores = pid2visit2question_sub_scores[p][vname]
                        if i in [0, 1]:
                            scores.sub_scores[0][ques] = s
                        else:
                            scores.sub_scores[i - 1][ques] = s
                    else:
                        score = pid2visit2part_sub_score[p][vname]
                        assert score[i] is None
                        score[i] = s

    summed_pid2visit2part_sub_score = defaultdict(dict)
    for p, v2scores in pid2visit2part_sub_score.items():
        for v, scores in v2scores.items():
            if scores[0] is None or scores[1] is None:
                summed_pid2visit2part_sub_score[p][v] = [None] + scores[2:]
            else:
                summed_pid2visit2part_sub_score[p][v] = [
                    scores[0] + scores[1]
                ] + scores[2:]

    return pid2visit2question_sub_scores, summed_pid2visit2part_sub_score


def _read_part_score(
    data_root: str,
    file_name: str,
    total_entry_name: str,
    include_evt_entry_list: tp.Optional[tp.List[str]],
) -> tp.Dict[str, tp.Dict[str, tp.Optional[float]]]:

    # read part1, part2, or part4 score from the file

    file_path = osp.join(data_root, file_name)
    df = pd.read_csv(file_path)

    patient_id_list = df["PATNO"].tolist()
    event_list = df["EVENT_ID"].tolist()
    total_list = np.array(df[total_entry_name].tolist()).astype(str)

    pid2evt2score: tp.Dict[str, tp.Dict[str, tp.Optional[float]]] = defaultdict(dict)
    for p, e, t in zip(patient_id_list, event_list, total_list):
        if include_evt_entry_list is not None and e not in include_evt_entry_list:
            continue
        pid2evt2score[str(p)][str(e)] = t

    return pid2evt2score


def _read_part3_score(
    data_root: str,
    file_name: str,
    total_entry_name: str,
    include_evt_entry_list: tp.Optional[tp.List[str]],
):
    # parse only ON scores

    file_path = osp.join(data_root, file_name)
    df = pd.read_csv(file_path)

    patient_id_list = df["PATNO"].tolist()
    event_list = df["EVENT_ID"].tolist()
    total_list = np.array(df[total_entry_name].tolist()).astype(str)
    pd_state_list = df["PDSTATE"].tolist()
    pag_name_list = df["PAG_NAME"].tolist()
    dbs_stat_list = df["DBS_STATUS"].tolist()

    pid2evt2item_list: tp.Dict[str, tp.Dict[str, tp.List]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p, e, t, pd_state, pag_name, dbs_stat in zip(
        patient_id_list,
        event_list,
        total_list,
        pd_state_list,
        pag_name_list,
        dbs_stat_list,
    ):
        pid2evt2item_list[str(p)][str(e)].append([t, pd_state, pag_name, dbs_stat])

    pid2evt2score: tp.Dict[str, tp.Dict[str, tp.Optional[float]]] = defaultdict(
        lambda: defaultdict(None)
    )
    for pid, evt2item in pid2evt2item_list.items():
        for evt, item_list in evt2item.items():

            if include_evt_entry_list is not None and evt not in include_evt_entry_list:
                continue

            # NOTE(YL 11/27):: parse ON part3 score
            state_list = [i[1] for i in item_list]
            dbs_list = [i[3] for i in item_list]
            if len(state_list) > 1:
                if "ON" in state_list:
                    pid2evt2score[pid][evt] = item_list[state_list.index("ON")][0]
                elif 1 in dbs_list:
                    assert 1 in dbs_list, "dbs_list: {}".format(dbs_list)
                    pid2evt2score[pid][evt] = item_list[dbs_list.index(1)][0]
                else:
                    # NOTE(YL 11/27):: corner case
                    if pid == "74199":
                        pid2evt2score[pid][evt] = item_list[0][0]
                    elif pid == "100889":
                        pid2evt2score[pid][evt] = item_list[0][0]
            else:
                pid2evt2score[pid][evt] = item_list[0][0]

    return pid2evt2score
