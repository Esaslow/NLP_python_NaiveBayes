import numpy as np
import os
import pandas as pd


def extract_features(mail_dir,body_line = 2, subject_line = 0):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    bodys = [];
    subjects = [];
    target = []
    for fil in files:
        if 'spmsg' in fil:
            target.append(1)
        else:
            target.append(0)
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == body_line:
                    bodys.append(line)
                if i == subject_line:
                    subjects.append(line)
    return bodys,subjects,target

def plot_feat_importance(ax,model,vectors,cutoff = .0001):
    inv_dict = {v: k for k, v in vectors.vocabulary_.items()}
    cols = []
    feat_scores = model.feature_importances_
    for i in range(len(feat_scores)):
        cols.append(inv_dict[i])
    feat_scores = pd.DataFrame({'Average Gini importance' : model.feature_importances_},
                               index=cols)


    scores = feat_scores.sort_values('Average Gini importance',ascending=False)
    #_,ax = plt.subplots(1,1,figsize = (20,15))
    scores2 = scores[scores['Average Gini importance'] > cutoff]
    scores2.sort_values('Average Gini importance').plot(kind = 'barh',ax = ax)
    ax.set_title('Subject line Average gini Importance');

    return ax
