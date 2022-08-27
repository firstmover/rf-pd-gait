#! /usr/bin/env Rscript
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 04/25/2021
#
# Distributed under terms of the MIT license.

library(ggeffects)
library(lmerTest)
library(argparse)
library(crayon)

parser <- ArgumentParser()

parser$add_argument(
  "--data_file",
  default =
    "./data/covariate_adjusted_lm/data.csv",
  help = "data file"
)
parser$add_argument(
  "--save_coefficient_path",
  default =
    "./data/covariate_adjusted_lm/coefficients",
  help = "save coefficient path"
)
parser$add_argument("-s", "--seed",
  type = "integer", default = 233,
  help = "Random seed [default %(default)s]",
  metavar = "number"
)

args <- parser$parse_args()

set.seed(args$seed)

data_path <- args$data_file
if (!file.exists(data_path)) {
  cat("File does not exist: ", data_path)
  quit()
}

data <- read.csv(data_path)

save_coefficient_path <- args$save_coefficient_path
dir.create(save_coefficient_path)

base_covariates <- list(
  "spd", "age", "is_female",
  "is_white", "is_hispanic_latino", "is_college_graduate"
)
confounding_factors <- list(
  "moca", "mcirs", "mcirs_1",
  "mcirs_2", "mcirs_3", "mcirs_4",
  "mcirs_5", "mcirs_6", "mcirs_7",
  "mcirs_8", "mcirs_9", "mcirs_10",
  "mcirs_11", "mcirs_13", "mcirs_14"
)

cat(cyan("## part3 model \n\n"))

formula <- paste(
  "part3 ~ ",
  paste(base_covariates, collapse = " + ")
)
cat(yellow(formula, "\n"))
model <- lm(
  formula,
  data = data,
)
model_summary <- summary(model)
print(model_summary)

dir.create(file.path(save_coefficient_path, "part3"))
path <- file.path(save_coefficient_path, "part3", "core.csv")
write.csv(model_summary$coefficients, path)

for (covar in confounding_factors) {
  formula <- paste(
    "part3 ~ ",
    paste(append(base_covariates, covar), collapse = " + ")
  )
  cat(yellow(formula, "\n"))
  model <- lm(
    formula,
    data = data,
  )

  model_summary <- summary(model)
  print(model_summary)

  path <- file.path(save_coefficient_path, "part3", paste(covar, ".csv", sep = ""))
  write.csv(model_summary$coefficients, path)
}

cat(cyan("## total model \n\n"))

formula <- paste(
  "total ~ ",
  paste(base_covariates, collapse = " + ")
)
cat(yellow(formula, "\n"))
model <- lm(
  formula,
  data = data,
)

model_summary <- summary(model)
print(model_summary)

dir.create(file.path(save_coefficient_path, "total"))
path <- file.path(save_coefficient_path, "total", "core.csv")
write.csv(model_summary$coefficients, path)

for (covar in confounding_factors) {
  formula <- paste(
    "total ~ ",
    paste(append(base_covariates, covar), collapse = " + ")
  )
  cat(yellow(formula, "\n"))
  model <- lm(
    formula,
    data = data,
  )

  model_summary <- summary(model)
  print(model_summary)

  path <- file.path(save_coefficient_path, "total", paste(covar, ".csv", sep = ""))
  write.csv(model_summary$coefficients, path)
}
