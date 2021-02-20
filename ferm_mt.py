import numpy as np
import time
import json
import argparse
import cvxopt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from measures import equalized_odds_measure_TP, eq_treatment_measure
from data_loader import load_dataset
from new_measures import FP_TP_measure


def linear_kernel(x, y):
  return np.dot(x, y.T)


def save_expt(fname, ckpt):
  with open(fname, 'w') as f:
    json.dump(ckpt,f, indent=2)


class FERM_MT(BaseEstimator):
  '''
  FERM extension that optimizes for
  Equal Treatment
  '''

  def __init__(self, kernel='rbf', C=1.0,
               sensible_feature=None,
               gamma=1.0,
               rho=0.0):
    self.kernel = kernel
    self.C = C
    self.fairness = False if sensible_feature is None else True
    self.sensible_feature = sensible_feature
    self.gamma = gamma
    self.rho = rho
    self.w = None

  def fit(self, X, y):
    def rbfk(x, y): return rbf_kernel(x, y, self.gamma)
    if self.kernel == 'rbf':
      self.fkernel = rbfk
    elif self.kernel == 'linear':
      self.fkernel = linear_kernel
    else:
      self.fkernel = rbfk

    if self.fairness:
      self.values_of_sensible_feature = list(set(self.sensible_feature))
      self.v0 = np.min(self.values_of_sensible_feature)
      self.v1 = np.max(self.values_of_sensible_feature)
      va = [i for i in range(len(y)) if y[i] == -1 and
            self.sensible_feature[i] == self.v0]
      vb = [i for i in range(len(y)) if y[i] == -1 and
            self.sensible_feature[i] == self.v1]
      # na, nb = len(va), len(vb)
      self.set_s1 = va
      self.set_s2 = vb
      # if na > nb:
        # self.set_s1 = va
        # self.set_s2 = [i for i in range(len(y)) if y[i] == -1
                       # and self.sensible_feature[i] == self.v1]
      # else:
        # self.set_s1 = vb
        # self.set_s2 = [i for i in range(len(y)) if y[i] == -1
                       # and self.sensible_feature[i] == self.v0]

      self.n_s1 = len(self.set_s1)
      self.n_s2 = len(self.set_s2)

    N, n_dims = X.shape
    K = self.fkernel(X, X)
    P = np.outer(y, y) * K
    q = -np.ones(N)
    G = np.vstack([np.eye(N) * -1, np.identity(N)])
    h = np.hstack([np.zeros(N), np.ones(N) * self.C])

    if self.fairness:
      tau = [(np.sum(K[self.set_s1, i]) / self.n_s1) - (np.sum(K[self.set_s2, i]) / self.n_s2)
             for i in range(len(y))]
      fline = y*tau
      G = np.vstack([G, np.diag(fline)*-1])
      h = np.hstack([h, np.ones(N)*self.rho])

      # fline = y * tau
      # fline = cvxopt.matrix(fline, (1, N), 'd')

      # A = cvxopt.matrix(np.vstack([y, fline]))
      # b = cvxopt.matrix([0.0, self.rho])
    # else:
      # A = cvxopt.matrix(y.astype(np.double), (1, N), 'd')
      # b = cvxopt.matrix(0.0)

    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(y.astype(np.double), (1, N), 'd')
    b = cvxopt.matrix(0.0)

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    print('Solver status: ', sol['status'])
    a = np.ravel(sol['x'])
    sv = a > 1e-7
    ind = np.arange(len(a))[sv]
    self.a = a[sv]
    self.sv = X[sv]
    self.sv_y = y[sv]

    self.b = 0
    for i in range(len(self.a)):
      self.b += self.sv_y[i]
      self.b -= np.sum(self.a * self.sv_y * K[ind[i], sv])
      self.b /= len(self.a)

    if self.kernel == 'linear_kernel':
      self.w = np.zeros(n_features)
      for n in range(len(self.a)):
        self.w += self.a[n] * self.sv_y[n] * self.sv[n]
    else:
      self.w = None

  def project(self, X):
    if self.w is not None:
      return np.dot(X, self.w) + self.b
    else:
      XSV = self.fkernel(X, self.sv)
      a_sv_y = np.multiply(self.a, self.sv_y)
      y_pred = [np.sum(np.multiply(np.multiply(
          self.a, self.sv_y), XSV[i, :])) for i in range(len(X))]
      return y_pred + self.b

  def predict(self, X):
    return np.sign(self.project(X))

  def score(self, X_test, y_test):
    predict = self.predict(X_test)
    acc = accuracy_score(y_test, predict)
    return acc


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='compas',
    choices=['adult', 'compas', 'drug', 'arrhythm'], help='Dataset')

  parser.add_argument('-r', '--rho', type=float, default=0.1,
    help='Value of mistreatment param')

  args = parser.parse_args()

  data_train, data_test, sens_feature = load_dataset(args.dataset)
  sensible_feature = sens_feature
  sensible_feature_values = list(set(data_train.sens))
  print('Different values of sens feature: ', sensible_feature_values)

  print('Grid search for SVM..')
  param_grid = [{
  'C': [0.1, 1, 10.0],
  'gamma': [0.1, 0.01],
  'kernel': ['rbf']
  }]

  svc = svm.SVC()
  clf = GridSearchCV(svc, param_grid, n_jobs=8)
  start = time.time()
  clf.fit(data_train.data, data_train.target)
  print(f'Fitting vanilla SVM took: {time.time() - start} s')
  print('Best SVM estimator: ', clf.best_estimator_)
  pred_svc = clf.predict(data_test.data)
  pred_train = clf.predict(data_train.data)

  acc_svc_test = accuracy_score(data_test.target, pred_svc)
  acc_svc_train = accuracy_score(data_train.target, pred_train)

  EO_test = equalized_odds_measure_TP(data_test, clf, [sensible_feature],
    ylabel=1, sens_feats=data_test.sens)
  EO_train = equalized_odds_measure_TP(data_train, clf, [sensible_feature],
    ylabel=1, sens_feats=data_train.sens)
  # ET_test = eq_treatment_measure(data_test, clf, [sensible_feature], sens_feats=data_test.sens)
  dEO_svc = np.abs(EO_test[sensible_feature][sensible_feature_values[0]] -
                            EO_test[sensible_feature][sensible_feature_values[1]])

  print(data_test.target.shape)
  print(data_test.sens.shape)

  P, FPR, FNR, tpr_sens, fpr_sens, fnr_sens = FP_TP_measure(data_test.target,
      pred_svc, sens=data_test.sens)

  dFPR_svc = np.abs(fpr_sens[0] - fpr_sens[1])
  print('SVC FPR sensitive: ', fpr_sens)
  print('SVC TPR sensitive: ', tpr_sens)


  # dET_svc = ET_test[sensible_feature]
  # dET_svc = np.abs(ET_test[sensible_feature][sensible_feature_values[0]] -
      # ET_test[sensible_feature][sensible_feature_values[1]])

  print('SVM Test Acc: ', acc_svc_test)

  print('DEO test: ', np.abs(EO_test[sensible_feature][sensible_feature_values[0]] -
                            EO_test[sensible_feature][sensible_feature_values[1]]))
  print('DEO train: ', np.abs(EO_train[sensible_feature][sensible_feature_values[0]] -
                             EO_train[sensible_feature][sensible_feature_values[1]]))
  print('DFPR test: ', dFPR_svc)

  print('Grid search for FERM_Mt...')
  ferm =FERM_MT(sensible_feature = data_train.sens, rho = args.rho)
  clf=GridSearchCV(ferm, param_grid, n_jobs = 4)
  start = time.time()
  clf.fit(data_train.data, data_train.target)
  print(f'Fitting FERM took: {time.time() - start} s')
  print('Best Estimator FERM_MT: ', clf.best_estimator_)
  pred_test=clf.predict(data_test.data)
  pred_train=clf.predict(data_train.data)

  acc_ferm_test=accuracy_score(data_test.target, pred_test)
  acc_ferm_train=accuracy_score(data_train.target, pred_train)

  EO_test=equalized_odds_measure_TP(data_test, clf, [sensible_feature],
    ylabel = 1, sens_feats = data_test.sens)
  EO_train=equalized_odds_measure_TP(data_train, clf, [sensible_feature],
    ylabel = 1, sens_feats = data_train.sens)
  # ET_test = eq_treatment_measure(data_test, clf, [sensible_feature], sens_feats=data_test.sens)

  print('FERM Acc: ', acc_ferm_test)
  print('DEO test:', np.abs(EO_test[sensible_feature][sensible_feature_values[0]] -
                            EO_test[sensible_feature][sensible_feature_values[1]]))
  print('DEO train:', np.abs(EO_train[sensible_feature][sensible_feature_values[0]] -
                             EO_train[sensible_feature][sensible_feature_values[1]]))

  dEO_ferm=np.abs(EO_test[sensible_feature][sensible_feature_values[0]] -
                            EO_test[sensible_feature][sensible_feature_values[1]])

  P, FPR, FNR, tpr_sens, fpr_sens, fnr_sens = FP_TP_measure(data_test.target,
      pred_test, sens=data_test.sens)

  dFPR_ferm = np.abs(fpr_sens[0] - fpr_sens[1])
  # dET_test_ferm = ET_test[sensible_feature]
  # dET_test_ferm = np.abs(ET_test[sensible_feature][sensible_feature_values[0]] -
      # ET_test[sensible_feature][sensible_feature_values[1]])
  print('FERM FPR sensitive: ', fpr_sens)
  print('FERM TPR sensitive: ', tpr_sens)
  print('DFPR test: ', dFPR_ferm)
  # print(fpr_sens)
  ckpt={
  'dataset': args.dataset,
  'rho': args.rho,
  'svm_test_acc': acc_svc_test,
  'ferm_test_acc': acc_ferm_test,
  'dEO_svc': dEO_svc,
  'dEO_ferm': dEO_ferm,
  'dFPR_svc': dFPR_svc,
  'dFPR_ferm': dFPR_ferm
  }
  save_expt(f'{args.dataset}_{args.rho}.json', ckpt)
