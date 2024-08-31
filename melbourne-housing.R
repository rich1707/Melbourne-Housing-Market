# Load packages ----

library(tidyverse)

library(patchwork)
library(scales)

library(tidymodels)
library(finetune)
library(embed)
library(vip)
library(bonsai)

library(rstatix)
library(DescTools)
library(moments)
library(bestNormalize)
library(modeest)

library(lightgbm)

conflicted::conflict_prefer(name = "filter", winner = "dplyr")
conflicted::conflict_prefer(name = "discard", winner = "purrr")
conflicted::conflict_prefer(name = "skewness", winner = "moments")
conflicted::conflict_prefer(name = "vi", winner = "vip")

options(scipen = 999)

# Read in data ----

melbourne <- read_csv("Melbourne_housing_FULL.csv") |> 
   janitor::clean_names()

# Missing values ----

melbourne <- drop_na(melbourne, price)

melbourne <- melbourne |> 
   mutate(across(where(is.character), \(.x) if_else(.x == "#N/A", NA, .x)))

melbourne <- melbourne |> 
   mutate(
      suburb = if_else(is.na(distance), "Fawkner", suburb),
      postcode = if_else(is.na(distance), 3060, postcode),
      distance = if_else(is.na(distance), 12.4, distance)
   )

melbourne <- melbourne |> 
   mutate(regionname = case_when(
      is.na(regionname) & suburb == "Fawkner" ~ "Northern Metropolitan",
      is.na(regionname) & suburb == "Footscray" ~ "Western Metropolitan",
      is.na(regionname) & suburb == "Camberwell" ~ "Southern Metropolitan",
      .default = regionname
   )) 

melbourne <- melbourne |> 
   mutate(council_area = case_when(
      is.na(council_area) & suburb == "Fawkner" ~ "Hume City Council",
      is.na(council_area) & suburb == "Footscray" ~ "Maribyrnong City Council",
      is.na(council_area) & suburb == "Camberwell" ~ "Boroondara City Council",
      .default = council_area 
   ))

# Impossible values ----

melbourne <- melbourne |> 
   select(-bedroom2)

melbourne <- melbourne |> 
   mutate(bathroom = na_if(bathroom, 0))

# Outliers ----

melbourne <- melbourne |> 
   mutate(bathroom = if_else(bathroom > rooms, rooms, bathroom))

melbourne <- melbourne |> 
   mutate(car = if_else(car > 5, 5, car))

melbourne <- melbourne |> 
   group_by(suburb) |> 
   mutate(year_built = if_else(year_built < 1800, mfv(year_built), year_built)) |> 
   ungroup()

melbourne <- melbourne |> 
   mutate(year_built = if_else(year_built > 2018, 2018, year_built))

# Location variables ----

melbourne <- melbourne |> 
   select(-landsize)

melbourne <- melbourne |> 
   mutate(building_area = na_if(building_area, 0))

melbourne <- melbourne |> 
   group_by(type, rooms) |> 
   mutate(med_building_area = median(building_area, na.rm = TRUE)) |> 
   ungroup() |> 
   mutate(difference = building_area - med_building_area) |>
   mutate(ratio = abs(difference) / med_building_area)

melbourne <- melbourne |> 
   mutate(building_area = if_else(
      condition = difference < 0 & ratio > 0.5,
      true = NA_real_,
      false = building_area
   ))

melbourne <- melbourne |> 
   mutate(building_area = if_else(
      condition = ratio > 3,
      true = NA,
      false = building_area
   )) |> 
   select(-med_building_area, -difference, -ratio)

# Feature engineering ----

melbourne <- melbourne |> 
   mutate(date = dmy(date))

melbourne <- melbourne |> 
   mutate(across( 
      everything(), 
      .fn = \(x) is.na(x), 
      .names = "{.col}_{.fn}"
   ))

melbourne <- melbourne |> 
   rename_with(
      .fn = \(x) str_replace(x, "_1", "_impute"),
      .cols = ends_with("_1") 
   )

melbourne <- melbourne |> 
   discard(\(x) all(x == FALSE))


# Partition data ----

set.seed(2024)

melb_split <- initial_split(melbourne, prop = 0.75)

melb_train <- training(melb_split)
melb_test <- testing(melb_split)

melb_k_flds <- vfold_cv(melb_train, v = 10)

# Preprocessing ----

melb_recipe <- melb_train |> 
   recipe(price ~ .) |> 
   step_rm(address) |> 
   step_mutate(postcode = as.character(postcode)) |> 
   step_string2factor(all_nominal_predictors()) |> 
   step_impute_knn(all_predictors(), neighbors = tune()) |> 
   step_lencode_mixed(
      c(suburb, seller_g, postcode, council_area),
      outcome = vars(price)
   )

# Model specification ----

melb_spec_gbm <- 
   boost_tree(
      mtry = tune(), trees = tune(), min_n = tune(), 
      tree_depth = tune(), learn_rate = tune(), 
      loss_reduction = tune()
   ) |> 
   set_mode("regression") |> 
   set_engine("lightgbm")

# Workflow ----

melb_wrkflw_gbm <- workflow() |> 
   add_recipe(melb_recipe) |> 
   add_model(melb_spec_gbm) 

# Parameters ----
   
params_gbm <- extract_parameter_set_dials(melb_wrkflw_gbm) |> 
   update(mtry = mtry(range = c(1, 12)))

# tune hypers and fit model ----

doParallel::registerDoParallel()

set.seed(2024)

tune_grid_gbm <- tune_race_anova(
   object = melb_wrkflw_gbm,
   resamples = melb_k_flds,
   grid = 35,
   metrics = metric_set(rsq, rmse),
   param_info = params_gbm,
   control = control_race(verbose = TRUE)
)

doParallel::stopImplicitCluster()  

rsq_gbm <- select_best(tune_grid_gbm, metric = "rsq")

melb_wrkflw_gbm <- finalize_workflow(melb_wrkflw_gbm, rsq_gbm)

# Predict on test data and evaluate ----

preds_gbm <- predict(model_gbm, new_data = melb_test)

preds_tbl_gbm <- melb_test |> 
   bind_cols(preds_gbm) |> 
   select(.pred, price) |> 
   mutate(residuals = price - .pred)

rmse(preds_tbl_gbm, .pred, price)

preds_tbl_gbm |> 
   mutate(across(.pred:price, \(x) log(x))) |> 
   rmse(.pred, price)

rsq(preds_tbl_gbm, .pred, price)





