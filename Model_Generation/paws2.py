import numpy as np 
import pandas as pd 
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import tree, metrics
import time
import matplotlib.pyplot as plt 
import synthetic_data
from scipy.integrate import solve_ivp

RANDOM_SEED = 1111 
POSITIVE_LABEL = 1
N_JOBS = 1 

# parameters for bagging classifier 
NUM_ESTIMATORS = 32
MAX_SAMPLES = 0.8 
MAX_FEATURES = .5

# parameters for data generation / L-V model 
TIMESTEP = 5
NUM_TIMESTEPS = 100
N = 25
M = 25
PTS_PER_SEC = 100
LEN_TRAJ = 50 

# verbose or not 
VERBOSE = 0 

# set a random seed 
np.random.seed(seed = RANDOM_SEED)

# -------------------- DATA GENERATION -----------------------------
# takes in the original densities for the prey, predator, and traps 
def one_timestep(prey_density, pred_density, trap_density): 
    for idx in range(TIMESTEP): 
        # trap and predator evolving 
        y0 = np.dstack((pred_density, trap_density))
        y0 = y0.flatten() 
        sol = solve_ivp(synthetic_data.spatial_dynamics_traps, y0=y0, t_span=[0, 1], t_eval=np.linspace(0, 1, PTS_PER_SEC), args=(N, M))
        sol_use = sol.y.reshape((N, M, 2, int(PTS_PER_SEC)))
        pred_density, trap_density = sol_use[:, :, 0, :], sol_use[:, :, 1, :]
        pred_density, trap_density = pred_density[:, :, -1], trap_density[:, :, -1]

        # predator and prey evolving 
        y0 = np.dstack((prey_density, pred_density))
        y0 = y0.flatten() 
        sol = solve_ivp(synthetic_data.spatial_dynamics_traps, y0=y0, t_span=[0, 1], t_eval=np.linspace(0, 1, PTS_PER_SEC), args=(N, M))
        sol_use = sol.y.reshape((N, M, 2, int(PTS_PER_SEC)))
        prey_density, pred_density = sol_use[:, :, 0, :], sol_use[:, :, 1, :]
        prey_density, pred_density = prey_density[:, :, -1], pred_density[:, :, -1]
    return prey_density, pred_density, trap_density 

# final should be n x m x timestep 
def data_generation(file1, file2): 
    # generate original prey, predator, and trap density 
    data_init = synthetic_data.generate().reshape(N, M, 2, PTS_PER_SEC * LEN_TRAJ)
    prey_density, pred_density = data_init[:, :, 0, -1], data_init[:, :, 1, -1]
    trap_density = np.random.binomial(1, 1/13, (N, M)) * 10 
    # save a n x m x timestep array of predator and trap densities over NUM_TIMESTEPS
    pred_densities = pred_density 
    traps = trap_density
    for _ in range(NUM_TIMESTEPS): 
        prey_density, pred_density, _ = one_timestep(prey_density, pred_density, trap_density)
        pred_densities = np.dstack((pred_densities, pred_density))
        trap_density = np.random.binomial(1, 1/2, (N, M)) * 10 
        traps = np.dstack((traps, trap_density))
    # save pred_densities and traps into file locations 
    np.save(file1, pred_densities)
    np.save(file2, traps)

# -------------------------- CREATE DATAFRAME -----------------------------
# np_array is (num_trials, timestamps, grid_width, grid_height) array    
# resulting X should be a dataframe of location, density, timestamp

# massive dataframe M, N, timesteps
# preds, traps, labels, patrol effort  => n timesteps 
# probabilities 

def create_dataframe(preds, traps):
    num_parks, M, N, timesteps = preds.shape 
    data_dict = {"Timestamps": [], "Label": [], "X": [], "Y": [], "Density": [], "Trap Effort": []}
    for i in range(num_parks): 
        park_preds, park_traps = preds[i], traps[i]
        for time in range(timesteps): 
            grid = park_preds[:, :, time]
            for row in range(len(grid)): 
                for col in range(len(grid[0])): 
                    data_dict["Timestamps"].append(time) 
                    if time > 0 and park_traps[row, col, time - 1] == 10: 
                        data_dict["Label"].append(np.random.binomial(1, park_preds[row, col, time - 1]/10))
                    else: 
                        data_dict["Label"].append(0)
                    # populate trap_effort 
                    data_dict["Trap Effort"].append(park_traps[row, col][:time].sum() / (10 * (time + 1)))
                    data_dict["X"].append(row)
                    data_dict["Y"].append(col)
                    data_dict["Density"].append(grid[row][col])
    return pd.DataFrame(data_dict)

def setup_data(df): 
    labels, effort, timestamps = df.loc[:,"Label"].values, df.loc[:, "Trap Effort"].values, df.loc[:,"Timestamps"].values
    df = df.drop(['Label', "Trap Effort", 'Timestamps'], axis=1)
    features = df.values
    return features, labels, effort, timestamps

# ------------------------ iWare (PREDICTIVE MODEL) -------------------------------
class iWare: 
    def __init__(self, num_classifiers=5): 
        self.num_classifiers = num_classifiers 

        # each classifier will be trained on filtered data, depending on the trap effort thresold
        self.classifiers = None 
        self.trap_effort_thresholds = None 
        self.weights = None         

    def get_base_estimator(self): 
        base_estimator = tree.DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=20)
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
            idx = np.where(np.logical_or(train_effort >= self.trap_effort_thresholds[i], train_y == POSITIVE_LABEL))[0]

            if i > 0 and self.trap_effort_thresholds[i] == self.trap_effort_thresholds[i-1]:
                print('threshold {} same as previous, value {}. skipping'.format(i, self.trap_effort_thresholds[i]))
                classifiers.append(None)
                continue

            if idx.size == 0:
                print('no training points found for classifier {}, threshold = {}'.format(i, self.trap_effort_thresholds[i]))
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

            print('classifier {}, threshold {}, num x {}'.format(i, self.trap_effort_thresholds[i], train_x_filter.shape))
            start_time = time.time()

            # fit training data
            classifier.fit(train_x_filter, train_y_filter)

            print('  train time: {:.2f} seconds, score: {:.5f}'.format(
                    time.time() - start_time,
                    classifier.score(train_x_filter, train_y_filter)))

            classifiers.append(classifier)

            print('-------------------------------------------')

        return classifiers

    def get_trap_effort_thresholds(self, train_effort):
        trap_effort_threshold_percentile = np.linspace(0, 100, self.num_classifiers, endpoint=False)
        trap_effort_thresholds = np.percentile(train_effort, trap_effort_threshold_percentile)
        print('percentiles {}'.format(trap_effort_threshold_percentile))
        print('patrol thresholds {}'.format(trap_effort_thresholds))
        return trap_effort_thresholds

    def train_iware(self, all_train_x, all_train_y, all_train_effort, use_balanced=False, nsplits=5):
        self.trap_effort_thresholds = self.get_trap_effort_thresholds(all_train_effort)

        print('shape x', all_train_x.shape)
        print('shape y', all_train_y.shape)
        print('shape train_effort', all_train_effort.shape)

        self.weights = self.get_vote_matrix() 

        print('-------------------------------------------')
        print('training classifiers with all train data')
        print('-------------------------------------------')
        
        self.classifiers = self.train_classifiers(all_train_x, all_train_y, all_train_effort, use_balanced)

    
    def train_test_split_by_year(self, features, labels, trap_effort, timestamps, timestamp):
        # full year of test data
        train_idx = np.where(timestamps < timestamp)[0]
        test_idx  = np.where(timestamps == timestamp)[0]

        train_x      = features[train_idx, :]
        train_y      = labels[train_idx]
        train_effort = trap_effort[train_idx]

        test_x      = features[test_idx, :]
        test_y      = labels[test_idx]
        test_effort = trap_effort[test_idx]

        print('train x, y', train_x.shape, train_y.shape)
        print('test x, y ', test_x.shape, test_y.shape)
        print('trap effort train, test ', train_effort.shape, test_effort.shape)

        return train_x, test_x, train_y, test_y, train_effort, test_effort 

    # returns probability of a positive label 
    def predict(self, test_x, test_y, test_effort):
        num_test = test_y.shape[0]
        weighted_predictions = np.zeros(num_test)
        all_predictions = np.zeros((num_test, self.num_classifiers))

        # compute the classification interval for each point 
        classification = np.zeros(num_test)
        for i in range(num_test): 
            smaller_thresholds = np.where(test_effort[i] > self.trap_effort_thresholds)[0]
            # trap effort might be lower than all available thresholds
            if len(smaller_thresholds) == 0: 
                classification[i] = 0 
            else: 
                classification[i] = smaller_thresholds[-1]
        classification = classification.astype(int)

        for i in range(self.num_classifiers): 
            if self.classifiers[i] is None: 
                print("Classifier {} is None; skipping".format(i))
                continue 
            
            curr_predictions = self.classifiers[i].predict_proba(test_x)
            curr_predictions = curr_predictions[:, 1] # probability of positive label
            all_predictions[:, i] = curr_predictions 
            multiplier = np.zeros(num_test)
            for j in range(num_test): 
                multiplier[j] = self.weights[classification[j]][i]
            weighted_predictions += np.multiply(curr_predictions, multiplier) 
        
        return weighted_predictions 

# ------------------------- TESTING FUNCTIONS --------------------------------

# ??? determine optimal probability threshold for a positive label
def determine_threshold(label, predict_test_pos_probs, num_thresholds=50):
    thresholds = np.linspace(0, 1, num_thresholds)
    f1 = np.zeros(thresholds.size)
    precision = np.zeros(thresholds.size)
    recall = np.zeros(thresholds.size)
    auprc = np.zeros(thresholds.size)

    for i in range(num_thresholds): 
        predict_labels = predict_test_pos_probs > thresholds[i]
        predict_labels = predict_labels.astype(int) 

        f1[i]        = metrics.f1_score(label, predict_labels)
        precision[i] = metrics.precision_score(label, predict_labels, pos_label=POSITIVE_LABEL)
        recall[i]    = metrics.recall_score(label, predict_labels, pos_label=POSITIVE_LABEL)

        precision_vals, recall_vals, _ = metrics.precision_recall_curve(label, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
        auprc[i]     = metrics.auc(recall_vals, precision_vals)
        
        if VERBOSE:
            print('threshold: {:.4f} | f1: {:.4f},  precision: {:.4f}, recall: {:.4f}, AUPRC: {:.4f}'.format(thresholds[i], f1[i], precision[i], recall[i], auprc[i]))

        # ??? why are we just using auprc ... ok i guess
        opt = np.argmax(f1)
        print('optimal threshold {:.4f}, with f1 {:.4f}, precision {:.4f}, recall {:.4f}, AUPRC {:.4f}'.format(thresholds[opt], f1[opt], precision[opt], recall[opt], auprc[opt]))

    return thresholds[opt]

def evaluate_results(test_y, predict_test_pos_probs): 
    output = []
 
    # compute optimal threshold and determine labels
    opt_threshold = determine_threshold(test_y, predict_test_pos_probs)
    print("PINEAPPLE", opt_threshold)
    predict_test = (predict_test_pos_probs > opt_threshold).astype(int)

    predict_test_neg_probs = np.ones(predict_test_pos_probs.shape) - predict_test_pos_probs
    predict_test_probs = np.concatenate((predict_test_neg_probs.reshape(1,-1), predict_test_pos_probs.reshape(1,-1)), axis=0).transpose()

    # select the prediction column with probability of assigned label
    predict_test_label_probs = predict_test_probs[[i for i in range(predict_test.shape[0])], tuple(predict_test)]

    fpr, tpr, _ = metrics.roc_curve(test_y, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
    output.append('AUC: {:.5f}'.format(metrics.auc(fpr, tpr)))

    precision_vals, recall_vals, _ = metrics.precision_recall_curve(test_y, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
    output.append('AUPRC: {:.5f}'.format(metrics.auc(recall_vals, precision_vals)))  # area under precision-recall curve
    #output.append('average precision score: {:.5f}'.format(metrics.average_precision_score(test_y, predict_test_pos_probs, pos_label=POSITIVE_LABEL)))

    output.append('precision: {:.5f}'.format(metrics.precision_score(test_y, predict_test, pos_label=POSITIVE_LABEL)))
    recall = metrics.recall_score(test_y, predict_test, pos_label=POSITIVE_LABEL)
    output.append('recall: {:.5f}'.format(recall))
    output.append('F1 score: {:.5f}'.format(metrics.f1_score(test_y, predict_test, pos_label=POSITIVE_LABEL)))

    percent_positive = np.where(predict_test == POSITIVE_LABEL)[0].shape[0] / predict_test.shape[0]
    l_and_l = recall ** 2 / percent_positive
    max_ll = 1 / (test_y.sum() / test_y.shape[0])
    output.append('L&L %: {:.5f} ({:.5f} / {:.5f})'.format(100 * (l_and_l / max_ll), l_and_l, max_ll))

    output.append('cross entropy: {:.5f}'.format(metrics.log_loss(test_y, predict_test_probs)))

    output.append('average prediction probability: {:.5f}'.format(predict_test_label_probs.mean()))
    output.append('accuracy: {:.5f}'.format(metrics.accuracy_score(test_y, predict_test)))
    output.append('cohen\'s kappa: {:.5f}'.format(metrics.cohen_kappa_score(test_y, predict_test)))  # measures inter-annotator agreement
    # output.append('F-beta score: {:.5f}'.format(metrics.fbeta_score(test_y, predict_test, 2, pos_label=POSITIVE_LABEL))) # commonly .5, 1, or 2. if 1, then same as f1

    return '\n'.join(output)


def run(): 
    file1 = "/Users/catherinecui/288_invasive_species/Data/paws_preds.npy"
    file2 = "/Users/catherinecui/288_invasive_species/Data/paws_traps.npy"
    # data_generation(file1, file2) 

    # shape is (M, N, NUM_TIMESTEPS)
    preds1, traps1 = np.load(file1), np.load(file2)
    preds1 = np.expand_dims(preds1, axis=0)
    traps1 = np.expand_dims(traps1, axis=0)
    df = create_dataframe(preds1, traps1)
    
    features, labels, effort, timestamps = setup_data(df)
    iware = iWare()
    train_x, test_x, train_y, test_y, train_effort, test_effort = iware.train_test_split_by_year(features, labels, effort, timestamps, 90)
    iware.train_iware(train_x, train_y, train_effort)

    predictions = iware.predict(test_x, test_y, test_effort)
    output = evaluate_results(test_y, predictions)
    print(output)

run()
