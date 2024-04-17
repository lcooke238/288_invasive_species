"""
PAWS dataset D
X: features, no current_patrol_effort 
    - setup_data in iware.py 
"""
"""
train 
"""

"""
iware (class iWare in iware.py): 
- trains an ensemble classifier from multiple weak learners (SVM, decision tree or GP?)
- ensemble classifier takes in a vector x(t, n) and predicts a binary label indicating whenter any illegal activity was detected at cell n during time t, can also call predict_proba to get probability of a label 
"""
import numpy as np 
import pandas as pd 
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import tree
import time

RANDOM_SEED = None 
N_JOBS = 1
MAX_SAMPLES = 0.8 
MAX_FEATURES = 0.5 
VERBOSE = 0
NUM_ESTIMATORS = 32
POSITIVE_LABEL = 1

# np_array is (num_trials, timestamps, grid_width, grid_height) array    
# resulting X should be a dataframe of location, density, timestamp, and patrol effort
def create_dataframe(density_arr, patrol_effort):
    (num, time, m, n) = density_arr.shape
    timestamps, density, x, y, effort = [], [], [], [], []
    for i in range(num): 
        times = density_arr[i]
        for j in range(time): 
            grid = times[j]
            for row in range(len(grid)): 
                for col in range(len(grid[0])): 
                    timestamps.append(j)
                    density.append(grid[row][col] > 0.5)
                    x.append(row)
                    y.append(col)
                    effort.append(patrol_effort[i][j][row][col])
    df = pd.DataFrame({"Timestamps": timestamps, "Density": density, "X": x, "Y": y, "Effort": effort})
    return df 

def setup_data(df): 
    labels, effort, timestamps = df.loc[:,"Density"].values, df.loc[:,"Effort"].values, df.loc[:,"Timestamps"].values
    df = df.drop(['Density', 'Effort', 'Timestamps'], axis=1)
    features = df.values
    return features, labels, effort, timestamps

def generate_data(num, time, m, n): 
    density = np.random.rand(num, time, m, n)
    effort = np.random.rand(num, time, m, n)
    return density, effort 

class iWare: 
    def __init__(self, method, num_classifiers): 
        self.method = method 
        self.num_classifiers = num_classifiers
        self.classifiers = None 
        self.patrol_thresholds = None 
        self.weights = None 
    
    def get_base_estimator(self): 
        if self.method == "dt": 
            base_estimator = tree.DecisionTreeClassifier(random_state=RANDOM_SEED)
        return base_estimator 
    
    def get_classifier(self, use_balanced): 
        base_estimator = self.get_base_estimator()
        if use_balanced:
            return BalancedBaggingClassifier(base_estimator=base_estimator,
                n_estimators=NUM_ESTIMATORS, max_samples=MAX_SAMPLES,
                max_features=MAX_FEATURES,
                bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False,
                sampling_strategy='majority', #sampling_strategy=0.8,
                replacement=True, n_jobs=N_JOBS,
                random_state=RANDOM_SEED, verbose=VERBOSE)

        # non-balanced bagging classifier used for other datasets
        else:
            return BaggingClassifier(base_estimator=base_estimator,
                n_estimators=NUM_ESTIMATORS, max_samples=MAX_SAMPLES,
                max_features=MAX_FEATURES,
                bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False, n_jobs=N_JOBS,
                random_state=RANDOM_SEED, verbose=VERBOSE)
    
    def get_vote_matrix(self):
        vote_power = np.identity(self.num_classifiers)                           # identity matrix
        vote_qual = np.ones((self.num_classifiers, self.num_classifiers))
        # create combined vote matrix
        vote_combine = np.multiply(vote_power, vote_qual)
        # normalize column-wise
        vote_combine = vote_combine / vote_combine.sum(1)[:,None]

        return vote_combine

    def train_classifiers(self, train_x, train_y, train_effort, use_balanced):
        classifiers = []
        for i in range(self.num_classifiers):
            #### filter data
            # get training data for this threshold
            idx = np.where(np.logical_or(train_effort >= self.patrol_thresholds[i], train_y == POSITIVE_LABEL))[0]

            if i > 0 and self.patrol_thresholds[i] == self.patrol_thresholds[i-1]:
                print('threshold {} same as previous, value {}. skipping'.format(i, self.patrol_thresholds[i]))
                classifiers.append(None)
                continue

            if idx.size == 0:
                print('no training points found for classifier {}, threshold = {}'.format(i, self.patrol_thresholds[i]))
                classifiers.append(None)
                continue
            train_x_filter = train_x[idx, :]
           
            print(train_y[250])
            train_y_filter = train_y[idx]

            print('filtered data: {}. num positive labels {}'.format(train_x_filter.shape, np.sum(train_y_filter)))

            if np.sum(train_y_filter) == 0:
                print('no positive labels in this subset of the training data. skipping classifier {}'.format(i))
                classifiers.append(None)
                continue

            # select and train a classifier
            classifier = self.get_classifier(use_balanced)

            print('classifier {}, threshold {}, num x {}'.format(i, self.patrol_thresholds[i], train_x_filter.shape))
            start_time = time.time()

            # fit training data
            classifier.fit(train_x_filter, train_y_filter)

            print('  train time: {:.2f} seconds, score: {:.5f}'.format(
                    time.time() - start_time,
                    classifier.score(train_x_filter, train_y_filter)))

            classifiers.append(classifier)

            print('-------------------------------------------')

        return classifiers

     # given a set of trained classifiers, compute optimal weights
    def get_classifier_weights(self, classifiers, reserve_x, reserve_y):
        # test classifiers on all data points
        predictions = []
        for i in range(self.num_classifiers):
            if classifiers[i] is None:
                predictions.append(np.zeros(reserve_y.shape))
                continue

            curr_predictions = classifiers[i].predict(reserve_x)
            predictions.append(curr_predictions)
        predictions = np.array(predictions).transpose()

        # define loss function
        def evaluate_ensemble(weights):
            # ensure we don't get NaN values
            if np.isnan(weights).any():
                return 1e9

            weighted_predictions = np.multiply(predictions, weights)
            weighted_predictions = np.sum(weighted_predictions, axis=1)

            score = metrics.log_loss(reserve_y, weighted_predictions)

            return score
            # evaluate score
            # auprc = metrics.average_precision_score(reserve_y, weighted_predictions, pos_label=POSITIVE_LABEL)
            #
            # # pass in negative to minimize
            # return -auprc

        # constrain weights to sum to 1
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        # bound weights to be between 0 and 1
        bounds = [(0,1)] * self.num_classifiers

        # random restarts with random initial weights
        #best_weights = np.ones(self.num_classifiers) / self.num_classifiers # default: equal weights
        best_weights = None
        best_score = 1e9

        # ensure we have enough positive samples
        unique_vals, unique_counts = np.unique(reserve_y, return_counts=True)
        unique_dict = dict(zip(unique_vals, unique_counts))
        if VERBOSE:
            print(unique_dict)
        if 1 not in unique_dict or unique_dict[1] < 5:
            print('  not enough positive labels. skipping')
            return best_weights

        for _ in range(10):
            w = np.random.rand(self.num_classifiers)
            w = w / np.sum(w)

            res = minimize(evaluate_ensemble, w, method='SLSQP', bounds=bounds, constraints=cons)
            if res.fun < best_score:
                best_weights = res.x
                best_score = res.fun

        if VERBOSE:
            print('best score {}, weights {}'.format(best_score, np.around(best_weights, 3)))
        return best_weights
    
    def get_patrol_thresholds(self, train_effort):
        patrol_threshold_percentile = np.linspace(0, 100, self.num_classifiers, endpoint=False)
        patrol_thresholds = np.percentile(train_effort, patrol_threshold_percentile)
        print('percentiles {}'.format(patrol_threshold_percentile))
        print('patrol thresholds {}'.format(patrol_thresholds))
        return patrol_thresholds

    def train_iware(self, all_train_x, all_train_y, all_train_effort, use_balanced=False, nsplits=5):
        self.patrol_thresholds = self.get_patrol_thresholds(all_train_effort)

        print('shape x', all_train_x.shape)
        print('shape y', all_train_y.shape)
        print('shape train_effort', all_train_effort.shape)
        # self.weights = np.eye(self.num_classifiers)
        self.weights = self.get_vote_matrix()

        print('-------------------------------------------')
        print('training classifiers with all train data')
        print('-------------------------------------------')

        self.classifiers = self.train_classifiers(all_train_x, all_train_y, all_train_effort, use_balanced)

    
    def train_test_split_by_year(self, features, labels, patrol_effort, timestamps, timestamp):
        # full year of test data
        train_idx = np.where(timestamps < timestamp)[0]
        test_idx  = np.where(timestamps == timestamp)[0]

        train_x      = features[train_idx, :]
        train_y      = labels[train_idx]
        train_effort = patrol_effort[train_idx]

        test_x      = features[test_idx, :]
        test_y      = labels[test_idx]
        test_effort = patrol_effort[test_idx]

        print('train x, y', train_x.shape, train_y.shape)
        print('test x, y ', test_x.shape, test_y.shape)
        print('patrol effort train, test ', train_effort.shape, test_effort.shape)

        return train_x, test_x, train_y, test_y, train_effort, test_effort

    
density, effort = generate_data(20, 50, 5, 5)
df = create_dataframe(density, effort)
print(df)
features, labels, effort, timestamps = setup_data(df)
print("Features: ", features)
print("Labels: ", labels)
print("Effort: ", effort)
print("Timestamps: ", timestamps)
iware = iWare("dt", 5)
train_x, test_x, train_y, test_y, train_effort, test_effort = iware.train_test_split_by_year(features, labels, effort, timestamps, 10)
iware.train_iware(train_x, train_y, train_effort)


# dataframe - add whether or not a trap was placed there 
# run decision tree on dataframe where time is < n - 1 to predict whether a trap would catch something if there was a trap there 
# place traps there 

# generating data 
# initialize random predators without traps and simulate distribution of predators 
# place random traps and use a variant of LV to generate timestamp 
# place traps at most dense locations 
# continute until we have 1000 timesteps 

