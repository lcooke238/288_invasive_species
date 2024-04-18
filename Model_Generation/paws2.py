import numpy as np 
import pandas as pd 
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import tree, metrics
import time
import matplotlib.pyplot as plt 

RANDOM_SEED = None 
POSITIVE_LABEL = 1

np.random.seed(seed = 12393)

# np_array is (num_trials, timestamps, grid_width, grid_height) array    
# resulting X should be a dataframe of location, density, timestamp
# need to feed in the traps 
def create_dataframe(density_arr, traps):
    (num, time, m, n) = density_arr.shape
    timestamps, caught, x, y, density = [], [], [], [], []
    for i in range(num): 
        times = density_arr[i]
        traps_per_time = traps[i]
        for j in range(time): 
            grid = times[j]
            for row in range(len(grid)): 
                for col in range(len(grid[0])): 
                    timestamps.append(j)
                    # note that we need to check if there was a trap placed there in the previous timestep
                    if j > 0 and traps_per_time[j][row][col] == 1: 
                        caught.append(np.random.binomial(1, grid[row][col]))
                    else: 
                        caught.append(0)
                    x.append(row)
                    y.append(col)
                    density.append(grid[row][col])
    df = pd.DataFrame({"Timestamps": timestamps, "Label": caught, "X": x, "Y": y, "Density": density})
    return df 

def setup_data(df): 
    labels, timestamps = df.loc[:,"Label"].values, df.loc[:,"Timestamps"].values
    df = df.drop(['Label', 'Timestamps'], axis=1)
    features = df.values
    return features, labels, timestamps

def generate_data(num, time, m, n): 
    density = np.random.rand(num, time, m, n)
    traps = np.random.binomial(1, 1/2, (num, time, m, n))
    return density, traps

class iWare: 
    def __init__(self, method): 
        self.method = method 
        self.base_estimator = None

    def train_iware(self, all_train_x, all_train_y):
        print('-------------------------------------------')
        print('training classifiers with all train data')
        print('-------------------------------------------')
        if self.method == "dt": 
            base_estimator = tree.DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=20)
            base_estimator.fit(all_train_x, all_train_y)
            self.base_estimator = base_estimator
        else: 
            raise Exception("Method not implemented.")


        """
        print('-------------------------------------------')
        print('printing decision tree')
        print('-------------------------------------------')

        plt.figure(figsize=(30,10), facecolor ='k')
        a = tree.plot_tree(base_estimator,

                   feature_names = ["X", "Y"],

                   class_names = ["0", "1"],

                   rounded = True,

                   filled = True,

                   fontsize=14)

        plt.show()
        """

    
    def train_test_split_by_year(self, features, labels, timestamps, timestamp):
        # full year of test data
        train_idx = np.where(timestamps < timestamp)[0]
        test_idx  = np.where(timestamps == timestamp)[0]

        train_x      = features[train_idx, :]
        train_y      = labels[train_idx]

        test_x      = features[test_idx, :]
        test_y      = labels[test_idx]

        print('train x, y', train_x.shape, train_y.shape)
        print('test x, y ', test_x.shape, test_y.shape)

        return train_x, test_x, train_y, test_y

    def predict(self, predict_x):
        if self.method == "dt": 
            prediction_probs = self.base_estimator.predict_proba(predict_x)
            predictions = self.base_estimator.predict(predict_x)
            return prediction_probs, predictions
        else: 
            raise Exception("Method not implemented.")


def evaluate_accuracy(y_pred, y): 
    return metrics.accuracy_score(y, y_pred)

density, traps = generate_data(20, 50, 5, 5)
df = create_dataframe(density, traps)
print(df)
features, labels, timestamps = setup_data(df)
print("Features: ", features)
print("Labels: ", labels)
print("Timestamps: ", timestamps)
iware = iWare("dt")
train_x, test_x, train_y, test_y = iware.train_test_split_by_year(features, labels, timestamps, 10)
iware.train_iware(train_x, train_y)
probs, predictions = iware.predict(test_x)
print(probs, predictions)
print(evaluate_accuracy(predictions, test_y))



# dataframe - add whether or not a trap was placed there 
# run decision tree on dataframe where time is < n - 1 to predict whether a trap would catch something if there was a trap there 
# place traps there 

# generating data 
# initialize random predators without traps and simulate distribution of predators 
# place random traps and use a variant of LV to generate timestamp 
# place traps at most dense locations 
# continute until we have 1000 timesteps 

