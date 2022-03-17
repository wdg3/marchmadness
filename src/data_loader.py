import pandas as pd
import glob

class DataLoader:
    def __init__(self, root_path, paths):
        self.root = root_path
        self.paths = []
        for path in paths:
            self.paths.append(self.root + path)
    
    def load_csvs(self):
        files = {}
        for path in self.paths:
            csvs = glob.glob(path + "*.csv")
            for csv in csvs:
                print("Reading " + csv + "...", end = "")
                files[csv.split('/')[-1]] = pd.read_csv(csv)
                print(" done.")
        
        return files