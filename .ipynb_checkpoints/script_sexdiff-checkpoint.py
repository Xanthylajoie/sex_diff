import glob
import numpy as np
import random
import pandas as pd
import seaborn as sns; sns.set()
import os
from tqdm import tqdm
from scipy.stats import zscore, norm, pearsonr
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting,image

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from pathlib import Path

from nltools.stats import fdr

from sklearn.metrics import confusion_matrix


def img4d2vector(img_path, masker):
    img_masked = masker.fit_transform(img_path)  #fait une moyenne par label
    return img_masked.flatten()  #devient 1 vecteur

def vector2img4d(vector, masker):
    data_2d = vector.reshape(8, -1)
    return masker.inverse_transform(data_2d)   #remettre dans espace MNI that we can plot with nilearn

#import data set
df = pd.read_csv("/home/xlajoie/Desktop/unrestricted_original.csv")

# Filter left-handers (>=50 handedness)
df = df[df["Handedness"] >= 50].reset_index()

# transform M and F with 0 and 1
df.rename(columns={"Subject": "subject_label"}, inplace=True)
df["Gender"] = df["Gender"].replace({"M": 0, "F": 1})


#save new csv with modifications as "final hcp database"
df.to_csv("/home/xlajoie/Desktop/Final_HCP_database.csv")

# load dataset containing sex and participant id
df = pd.read_csv("/home/xlajoie/Desktop/Final_HCP_database.csv")


df_boot = pd.read_csv("/home/xlajoie/Desktop/Final_HCP_database.csv")


df_bootstrap = pd.DataFrame()
bootstrap_coef = {}
model_list = {}

for i in range(0, 1000):
    for j in range(0, len(df_boot)):
        index = random.randint(0, len(df_boot)-1)
        frames = [df_bootstrap, df_boot[index:index+1]]
        df_bootstrap = pd.concat(frames)

    df_bootstrap = df_bootstrap.drop(df_bootstrap.columns[0], axis=1)

    atlas_dest = datasets.fetch_atlas_destrieux_2009()
    masker = NiftiLabelsMasker(atlas_dest.maps)

    img_tpl = "/data/brambati/dataset/HCP/derivatives/seed-to-voxel-nilearn/results/sub-{0}/sub-{0}_voxelcorrelations.nii.gz"  # à changer
    x_correl = []
    nb_subjects = len(df_bootstrap)
    subject_label = df_bootstrap["subject_label"][:nb_subjects]

    for sub in tqdm(subject_label):
        img_path = str(Path(img_tpl.format(sub)))  # format = remplace entre accolades par # sujet
        x_correl.append(img4d2vector(img_path, masker))

    x_correl = np.array(x_correl)
    x_correl.shape  # autant de lignes que de sujets, autant de colones (nb region atlas x 8 (seeds))

    y_sex = df_bootstrap["Gender"][:nb_subjects]  # maybe list(df["Gender"])

    print(sum(y_sex), len(y_sex))  #double check

    # split the sample o training/test with a 80/20 % ratio
    # and stratify sex by class, also shuffle the data

    X_train, X_test, y_train, y_test = train_test_split(
                                                        x_correl,  # x
                                                        y_sex,       # y
                                                        test_size = 0.2, # 80%/20% split
                                                        shuffle = True,  #shuffle dataset before splitting
                                                        stratify = y_sex,  # keep distribution of sex_class consistent between train and test sets
                                                        random_state = 123) #same shuffle each time

    print('train:', len(X_train),'test:', len(X_test))

    score = []
    model = LinearSVC()
    #score = cross_val_score(model, X_train, y_train, cv=10)
    score.append(cross_val_score(model, X_train, y_train, cv=10, n_jobs = 3).mean())
    print(score)

    model.fit(X_train,y_train) #fit the model/ train the model
    y_pred = model.predict(X_test)

    #calculate the model accuracy
    acc_test = model.score(X_test, y_test)
    acc_train = model.score(X_train, y_train)

    #compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    df_result = pd.DataFrame(data=[[tp, fn],[fp,tn]], index= ["Femme", "Homme"], columns = ["Femme", "Homme"])
    df_result = df_result.rename_axis("actual class")
    df_result = df_result.rename_axis("predicted class", axis="columns")
    print(df_result)

    # print results
    print('accuracy test (r2) =', acc_test)
    print('accuracy train (r2) =', acc_train)

    coef = model.coef_[0]
    bootstrap_coef[i] = coef
    model_list[i] = model
    df_bootstrap = pd.DataFrame()
    score = []
