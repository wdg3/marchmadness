import pandas as pd
from autogluon.tabular import TabularPredictor
from src.data_loader import DataLoader
from src.data_transformer import DataTransformer
from src.prediction_utils import *
from src.constants import *

data_path = "./data/"
file_path = "MDataFiles_Stage2/"

data_loader = DataLoader(data_path, [file_path])

files = data_loader.load_csvs()
transformer = DataTransformer(files, label="label")

train = transformer.get_train()
test = transformer.get_test()

predictor = TabularPredictor(label="label", eval_metric="log_loss").fit(train.drop(["TeamID_team1", "TeamID_team2"],axis=1))

y_pred = predictor.predict(test.drop(["TeamID_team1", "TeamID_team2"],axis=1))
probs = predictor.predict_proba(test.drop(["TeamID_team1", "TeamID_team2"],axis=1))

test["pred"] = y_pred
test["proba"] = probs[1]
names = files["MTeams.csv"]
test_with_names = test.merge(names[["TeamID", "TeamName"]], left_on="TeamID_team1", right_on="TeamID", suffixes=("", "_team1"))
test_with_names = test_with_names.merge(names[["TeamID", "TeamName"]], left_on="TeamID_team2", right_on="TeamID", suffixes=("", "_team2"))

pred_df = predict_probs_and_moneylines(test_with_names)

pred_df.to_csv("predictions/predictions.csv")

pretty_print_matchups(pred_df, first_four, include_moneyline=False)
pretty_print_matchups(pred_df, round_1_matchups, include_moneyline=False)