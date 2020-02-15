import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots

def bar_chart(df, feature, target):
    '''
    Plot stacked-bar to show relations between categorical feature and target 
    '''
    pos = df[df[target]==1][feature].value_counts()
    neg = df[df[target]==0][feature].value_counts()
    df = pd.DataFrame([pos,neg])
    df.index = ['Positive','Negative']
    df.plot(kind='bar', stacked=True, figsize=(10,5))


def kde_chart(df, feature, target, xlim=None):
    '''
    Plot kde to show relations between continuous num and target
    '''
    facet = sns.FacetGrid(df, hue=target, aspect=4)
    facet.map(sns.kdeplot, feature, shade=True)
    facet.set(xlim=(0, df[feature].max()))
    facet.add_legend()
    
    plt.show() if not xlim else plt.xlim(*xlim)


def plotImp(model, X, num = 30, lib='catboost'):
    feature_score = pd.DataFrame(
        list(zip(
            X.dtypes.index, 
            model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices))
        )),
        columns=['Feature','Score']
    )
    feature_score = feature_score.sort_values(
        by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')[:num]
    
    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
    ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
    ax.set_xlabel('')
    plt.show()