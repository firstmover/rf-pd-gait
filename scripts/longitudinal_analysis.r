#! /usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 02/23/2021
#
# Distributed under terms of the MIT license.

library(ggeffects)
library(lmerTest)
library(argparse)
library(crayon)

parser <- ArgumentParser()

parser$add_argument(
  "--hc_data_file",
  default = "./data/longitudinal_data_linear_regression/hc.csv",
  help = "data file"
)

parser$add_argument(
  "--pd_data_file",
  default = "./data/longitudinal_data_linear_regression/pd.csv",
  help = "data file"
)

parser$add_argument("-s", "--seed",
  type = "integer", default = 233,
  help = "Random seed [default %(default)s]",
  metavar = "number"
)

args <- parser$parse_args()

set.seed(args$seed)

pd_data_path <- args$pd_data_file
if (!file.exists(pd_data_path)) {
  cat("File does not exist: ", pd_data_path)
  quit()
}

hc_data_path <- args$hc_data_file
if (!file.exists(hc_data_path)) {
  cat("File does not exist: ", hc_data_path)
  quit()
}

pd_data <- read.csv(pd_data_path)
hc_data <- read.csv(hc_data_path)

########
#  hc  #
########

cat(cyan("# HC\n\n"))

data <- hc_data
data_path <- hc_data_path

data$delta_v <- data$v_mean - data$baseline_v_mean

cat(cyan("## HC full \n\n"))
m <- lmer(
  delta_v ~ x_month + age + is_female + is_hispanic_latino + is_college_graduate + (1 + x_month || device_id) - 1,
  data = data,
)
summary(m)

cat(cyan("## HC reduced \n\n"))
m_reduced <- lmer(
  delta_v ~ x_month + (1 + x_month || device_id) - 1,
  data = data,
)
summary(m_reduced)

cat(cyan("## partial f test \n\n"))
anova(m, m_reduced)

m_pred <- ggpredict(m_reduced, terms = "x_month [0:24 by=1]")
m_pred
save_path <- sub(".csv", "_pred.csv", data_path)
write.csv(m_pred, save_path)

cat(cyan("## fix effect \n\n"))
fixed_effect <- fixef(m_reduced)
fixed_effect
save_path <- sub(".csv", "_fixed_effect.csv", data_path)
write.csv(fixed_effect, save_path)

cat(cyan("## random effect \n\n"))
rand_effect <- ranef(m_reduced)
rand_effect
save_path <- sub(".csv", "_random_effect.csv", data_path)
write.csv(rand_effect, save_path)

########
#  pd  #
########

cat(cyan("# PD\n\n"))

data <- pd_data

data$delta_v <- data$v_mean - data$baseline_v_mean
data_path <- pd_data_path

cat(cyan("## PD full \n\n"))

m <- lmer(
  delta_v ~ x_month + age + is_female + is_college_graduate + (1 + x_month || device_id) - 1,
  data = data,
)
summary(m)

cat(cyan("## PD reduced\n\n"))
m_reduced <- lmer(
  delta_v ~ x_month + (1 + x_month || device_id) - 1,
  data = data,
)
summary(m_reduced)

cat(cyan("## partial f test\n\n"))
anova(m, m_reduced)

m_pred <- ggpredict(m_reduced, terms = "x_month [0:24 by=1]")
m_pred
save_path <- sub(".csv", "_pred.csv", data_path)
write.csv(m_pred, save_path)

cat(cyan("## fix effect \n\n"))
fixed_effect <- fixef(m_reduced)
fixed_effect
save_path <- sub(".csv", "_fixed_effect.csv", data_path)
write.csv(fixed_effect, save_path)

cat(cyan("## random effect \n\n"))
rand_effect <- ranef(m_reduced)
rand_effect
save_path <- sub(".csv", "_random_effect.csv", data_path)
write.csv(rand_effect, save_path)

#############
#  pd + hc  #
#############

cat(cyan("# HC + PD\n\n"))

data <- rbind(pd_data, hc_data)

data$delta_v <- data$v_mean - data$baseline_v_mean

cat(cyan("## full\n\n"))

m <- lmer(
  delta_v ~ x_month + x_month * is_pd + age + is_female + is_hispanic_latino + is_college_graduate + (1 + x_month || device_id) - 1,
  data = data,
)
summary(m)

cat(cyan("## partial\n\n"))

m_reduced <- lmer(
  delta_v ~ x_month + x_month * is_pd + (1 + x_month || device_id) - 1,
  data = data,
)
summary(m_reduced)

cat(cyan("## partial f test\n\n"))
anova(m, m_reduced)
