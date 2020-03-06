import pandas as pd
import sklearn.metrics as metrics

def missing(df, excluding=None):
    df = df if not excluding else df[df.columns[~df.columns.isin(excluding)]]
    total = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum()/df.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    return pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

def value_counts(df, fields, dropna=False, transpose=False):
    def print_f(f):
        vc = df[f].value_counts
        f_builder = pd.concat(
            (
                vc(dropna=dropna).rename('Count'),
                vc(dropna=dropna, normalize=True).round(3).rename('%'),
            ),
            axis='columns',
        )
        f_builder = f_builder.rename_axis(f)
        return f_builder.transpose() if transpose else f_builder

    if type(fields) == type(''):
        return print_f(fields)
    elif type(fields) == type([]):  
#         return pd.concat([print_f(f) for f in fields], keys=fields)
        return [print_f(f) for f in fields]
    else:
        print('Error: input fields should be string or list of strings')

def metric_summary(y_true, y_score, threshold=0.5):
    result = {}
    y_score = (y_score > threshold).astype(int)
    
    n_r = 3
    result['threshold'] = round(threshold, n_r)
    result['precision'] = round(metrics.precision_score(y_true, y_score), n_r)
    result['recall'] = round(metrics.recall_score(y_true, y_score), n_r)
    result['f1'] = round(metrics.f1_score(y_true, y_score), n_r)
    
    return result