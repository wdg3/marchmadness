import pandas as pd

def break_even_moneyline(probability):
    if probability > 0.5:
        x = -(100 / (1-probability))+100
    else:
        x = 100/probability - 100
    return x

def predict_probs_and_moneylines(data):
    predictions = pd.DataFrame(columns=["Team1", "Seed1", "Team2", "Seed2", "Win%1", "Win%2", "ML1", "ML2"])
    pred_array = []
    for i in range(len(data)):
        game = data[["TeamName", "TeamName_team2", "Seed_team1", "Seed_team2", "pred", "proba"]].iloc[i]
        mirror = data[(data["TeamName"] == game["TeamName_team2"]) & (data["TeamName_team2"] == game["TeamName"])]
        prob1 = float((game["proba"] + (1 - mirror["proba"])) / 2)
        prob2 = float(((1-game["proba"]) + mirror["proba"]) / 2)
        pred_array.append([game["TeamName"], game["Seed_team1"], game["TeamName_team2"], game["Seed_team2"], prob1, prob2, int(break_even_moneyline(prob1)), int(break_even_moneyline(prob2))])
    
    return pd.DataFrame(pred_array, columns = predictions.columns)

def pretty_print_matchups(pred_df, matchups, include_moneyline=True):
    for (x, y) in matchups:
        game = pred_df[(pred_df["Team1"] == x) & (pred_df["Team2"]==y)].iloc[0]
        print(game["Team1"] + " (" + str(game["Seed1"]) + ") vs. " + game["Team2"] + " (" + str(game["Seed2"]) + "):")
        print("{:.2%}".format(game["Win%1"]) + " : {:.2%}".format(game["Win%2"]))
        if (include_moneyline):
            print("Moneyline: " + str(abs(game["ML1"])))
        print()