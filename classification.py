import numpy as np
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from data import load_data, data_preprocessing

ALGORITHMS = {'Naive_Bayes' : GaussianNB(),
              # 'Decision_Tree' : DecisionTreeClassifier(),
              # 'K-NN' : KNeighborsClassifier(),
              # 'MLP' : MLPClassifier(solver='lbfgs', alpha=1e-5,
              #                       hidden_layer_sizes=(5,2), random_state=1),
              'Random_Forest' : RandomForestClassifier()}
              # 'Logistic_Regression' : LogisticRegression(),
              # 'Ada_Boost' : AdaBoostClassifier()}
              # 'SVM' : SVC()}

NB_FOLD = 10

def main():

    train, test = load_data()
    X, y, x_test = data_preprocessing(train, test)
    print(X)

    kf = KFold(n_splits=NB_FOLD)

    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        print("ITERATION {}:".format(idx))
        for name, alg in ALGORITHMS.items():
            print("\t{}".format(name))
            x_train = X[train_index]
            x_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            alg.fit(x_train, y_train)
            y_pred = alg.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            (prec, recall, f1, sup) = precision_recall_fscore_support(y_test,
                                                                      y_pred,
                                                                      average='macro')
            print("\t\tAccuracy: {}".format(acc))
            print("\t\tPrecision: {}".format(prec))
            print("\t\tRecall: {}".format(recall))
            print("\t\tF1: {}".format(f1))
            print("\t\tLog loss: {}".format(loss))


if __name__ == '__main__':
    main()
