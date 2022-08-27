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
from typing import Dict, List, Optional


class MDSUPDRSSubScore:

    sub_score_name_list = [
        [
            "Cognitive Impairment",
            "Hallucinations and Psychosis",
            "Depressed Mood",
            "Anxious Mood",
            "Apathy",
            "Dopamine Dysregulation",
            "Sleep Problems",
            "Daytime Sleepiness",
            "Pain and Other Sensations",
            "Urinary Problems",
            "Constipation Problems",
            "Light Headedness on Standing",
            "Fatigue",
        ],
        [
            "Speech",
            "Salive and Drooling",
            "Chewing and Swallowing",
            "Eating Tasks",
            "Dressing",
            "Hygiene",
            "Handwriting",
            "Doing Hobbies and Other Activities",
            "Turning in Bed",
            "Tremor",
            "Getting Out of Bed, a Car, or a Deep Chair",
            "Walking and Balance",
            "Freezing",
        ],
        [
            "Is Taking Medication",
            "Speech",
            "Facial Expression",
            "Rigidity - Neck",
            "Rigidity - RUE",
            "Rigidity - LUE",
            "Rigiditiy - RLE",
            "Rigidity - LLE",
            "Finger Tapping - Right",
            "Finger Tapping - Left",
            "Hand Movements - Right",
            "Hand Movements - Left",
            "Hand Pronate-Suprinate - Right",
            "Hand Pronate-Suprinate - Left",
            "Toe Tapping - Right",
            "Toe Tapping - Left",
            "Leg Agility - Right",
            "Leg Agility - Left",
            "Arising from Chair",
            "Gait",
            "Freezing of Gait",
            "Postural Stability",
            "Posture",
            "Body Bradykinesia",
            "Postural Hand Tremor - Right",
            "Postural Hand Tremor - Left",
            "Kinetic Hand Tremor - Right",
            "Kinetic Hand Tremor - Left",
            "Rest Tremor Amplitude - RUE",
            "Rest Tremor Amplitude - LUE",
            "Rest Tremor Amplitude - RLE",
            "Rest Tremor Amplitude - LLE",
            "Rest Tremor Amplitude - Lip/Jaw",
            "Constancy of Rest Tremor",
        ],
        [
            "Time w/ Dyskinesias",
            "Total Hours Awake",
            "Total Hours w/ Dyskinesia",
            "% Dyskinesia",
            "Function Impact of Dyskinesias",
            "Time spent in the OFF State",
            "Total Hours Awake",
            "Total Hours OFF",
            "% OFF",
            "Functional Impact of Fluctuations",
            "Complexity of Motor Fluctuations",
            "Painful OFF-state dystonia",
            "Total Hours OFF",
            "Total Hours OFF w/ Dystonia",
            "% OFF Dystonia",
        ],
    ]

    def __init__(self, raw_scores_list: Optional[List[Dict[str, float]]] = None):

        self.sub_scores: List[Dict[str, Optional[float]]] = []

        if raw_scores_list is not None:
            for name_list, raw_score in zip(self.sub_score_name_list, raw_scores_list):
                self.sub_scores.append({n: raw_score[n] for n in name_list})
        else:
            for name_list in self.sub_score_name_list:
                self.sub_scores.append({n: None for n in name_list})

    @property
    def non_tremor_score(self):
        name_list = self.sub_score_name_list[2][:23]
        return sum([self.sub_scores[2][n] for n in name_list])

    @property
    def tremor_score(self):
        name_list = self.sub_score_name_list[2][:23]
        return sum(
            [
                self.sub_scores[2][n]
                for n in self.sub_score_name_list[2]
                if n not in name_list
            ]
        )

    @property
    def rigidity_score(self):
        name_list = self.sub_score_name_list[2][2:7]
        return sum([self.sub_scores[2][n] for n in name_list])

    @property
    def bradykinesia_score(self):
        name_list = (
            self.sub_score_name_list[2][1:2]
            + self.sub_score_name_list[2][7:18]
            + self.sub_score_name_list[2][22:23]
        )
        return sum([self.sub_scores[2][n] for n in name_list])

    @property
    def axial_score(self):
        name_list = (
            self.sub_score_name_list[2][0:1] + self.sub_score_name_list[2][17:22]
        )
        return sum([self.sub_scores[2][n] for n in name_list])
