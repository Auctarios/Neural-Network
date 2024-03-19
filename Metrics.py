from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score
from sklearn.metrics import precision_score, recall_score, hamming_loss
import numpy as np

class Metrics:
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list

    def calc(self, y_pred, y_true, problem_type='binary', average_method='macro'):
        results = self._calculate_metrics(y_pred, y_true, problem_type, average_method)
        
        # Format the results into a string
        results_str = ', '.join([f"{metric}: {value:.4f}" for metric, value in results.items()])
        
        return results_str, results

    def _calculate_metrics(self, y_pred, y_true, problem_type, average_method):
        results = {}
        for metric in self.metrics_list:
            if problem_type in ['binary', 'multiclass']:
                if metric == 'acc':
                    #
                    results['acc'] = accuracy_score(y_true, np.round(y_pred)) if problem_type == 'binary' else accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
                if metric == 'auc' and problem_type == 'binary':
                    #
                    results['auc'] = roc_auc_score(y_true, y_pred)
                if metric in ['precision', 'recall', 'f1']:
                    #
                    func = {'precision': precision_score, 'recall': recall_score, 'f1': f1_score}[metric]
                    results[metric] = func(y_true, np.round(y_pred),average=average_method) if problem_type == 'binary' else func(np.argmax(y_true, axis=1), np.argmax(y_pred,axis=1),average=average_method)
            elif problem_type == 'multilabel':
                if metric == 'hamming_loss':
                    #
                    results['hamming_loss'] = hamming_loss(y_true, np.round(y_pred))
                if metric in ['precision', 'recall', 'f1']:
                    func = {'precision': precision_score, 'recall': recall_score, 'f1': f1_score}[metric]
                    results[metric] = func(y_true, np.round(y_pred), average=average_method)
            elif problem_type == 'regression':
                if metric == 'mse':
                    results['mse'] = mean_squared_error(y_true, y_pred)
                if metric == 'rmse':
                    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        return results
