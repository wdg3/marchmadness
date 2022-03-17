from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = len(train_data)
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
save_path = "../models"

label = 'class'
print("Summary of class variable: \n", train_data[label].describe())

predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label]
test_data_nolab = test_data.drop(columns=[label])
test_data_nolab.head()

predictor = TabularPredictor.load(save_path)
y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

predictor.leaderboard(test_data, silent=True)
