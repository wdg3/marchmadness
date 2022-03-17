import numpy as np
import pandas as pd

class DataTransformer:
    def __init__(self, dataframes, label):
        self.dfs = dataframes
        self.rankings = self.transform_rankings(self.dfs)
        self.train = self.rankings[self.rankings["Season"] != 2022]
        self.test = self.rankings[self.rankings["Season"] == 2022]
        self.train = self.add_labels(self.train, label)
        self.test = self.process_test(self.test)
    
    def transform_rankings(self, data):
        print("Transforming historical rankings data...", end="")
        ordinals = data['MMasseyOrdinals.csv']
        seasons = np.unique(ordinals["Season"])
        systems = np.unique(ordinals["SystemName"])
        final_ordinals = pd.DataFrame()
        all_finals = []
        for season in seasons:
            season_frame = ordinals.loc[ordinals["Season"] == season]
            for system in systems:
                season_system_frame = season_frame.loc[season_frame["SystemName"] == system]
                if not (season_system_frame.empty):
                    maximum_day = max(season_system_frame["RankingDayNum"])
                    season_system_finals = season_system_frame.loc[season_system_frame["RankingDayNum"] == maximum_day]
                    all_finals.append(season_system_finals)
        final_ordinals = pd.concat(all_finals, axis = 0)
        system_dfs = []
        for system in systems:
            system_dfs.append(final_ordinals.loc[final_ordinals["SystemName"] == system].drop(["SystemName", "RankingDayNum"], axis=1).rename(columns={"OrdinalRank": system}))
        joint_ordinals = system_dfs[0]
        for df in system_dfs[1:]:
            joint_ordinals = joint_ordinals.merge(df, how="outer", on=["Season", "TeamID"])
        seeds = data["MNCAATourneySeeds.csv"]
        confs = data["MTeamConferences.csv"]
        with_seeds = joint_ordinals.merge(seeds, how="inner", on=["Season", "TeamID"])
        with_conf = with_seeds.merge(confs, on=["Season", "TeamID"])
        data = with_conf
        data["Seed"] = data["Seed"].map(lambda x: int(x) if len(x) == 2 else (int(x[1:3]) if len(x) == 4 else int(x[1:])))

        print("done.")
        return data
        
    def add_labels(self, data, label):
        print("Labeling historical data...", end="")
        results = self.dfs["MNCAATourneyCompactResults.csv"].drop(["DayNum", "WScore", "LScore", "WLoc", "NumOT"], axis = 1)
        X = []
        y = []
        for i in range(len(results)):
            result = results.iloc[i]
            team1 = min(result["WTeamID"], result["LTeamID"])
            team2 = max(result["WTeamID"], result["LTeamID"])
            season = result["Season"]
            season_data = data.loc[data["Season"] == season]
            x1 = season_data.loc[(season_data["TeamID"] == team1)]
            x2 = season_data.loc[(season_data["TeamID"] == team2)]

            # We can make our data robust to which team is considered team1 in the input
            # by making a second input with the teams swapped
            if (x1.shape[0] == 1) and (x2.shape[0] == 1):
                x = x1.merge(x2, on=["Season"], suffixes=("_team1", "_team2"))
                xr = x2.merge(x1, on=["Season"], suffixes=("_team1", "_team2"))
                X.append(x)
                X.append(xr)
                if (team1 == result["WTeamID"]):
                    y.append(1)
                    y.append(0)
                else:
                    y.append(0)
                    y.append(1)

        y = pd.Series(y, name=label, dtype=int)
        X = pd.concat(X, axis=0)
        X = X.reset_index()
        X[label] = y

        # Some validation data tinkering suggested the best way to handle missing rankings
        # is to give teams the ~lowest possible rankings. For years where an entire column is
        # missing we set it constant, for systems that don't rank all teams, we treat unranked teams
        # as bad which seems reasonable.
        X = X.fillna(350)

        print("done.")
        return X
        
    def process_test(self, data):
        print("Generating possible tournament matchups for 2022...", end="")
        X = []
        for i in range(len(data)):
            for j in range(len(data)):
                if (i != j):
                    x1 = data.iloc[i].to_frame().T
                    x2 = data.iloc[j].to_frame().T
                    x = pd.merge(x1, x2, on=["Season"], suffixes=("_team1", "_team2"))
                    X.append(x)
        X = pd.concat(X, axis=0)
        X = X.reset_index()
        X = X.fillna(350)

        print("done.")
        
        return X
    
    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test