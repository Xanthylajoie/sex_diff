"""
"""

import numpy as np
import pandas as pd
import random

from joblib import Parallel, delayed, dump

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC


def img4d2vector(img_path, masker):
    img_masked = masker.fit_transform(img_path)  # fait une moyenne par label
    return img_masked.flatten()  # devient 1 vecteur


def run(df_boot, data):
    result = {}

    df_bootstrap = pd.DataFrame()

    for j in range(0, len(df_boot)):
        index = random.randint(0, len(df_boot) - 1)
        frames = [df_bootstrap, df_boot[index : index + 1]]
        df_bootstrap = pd.concat(frames)
    df_bootstrap = df_bootstrap.drop(df_bootstrap.columns[0], axis=1)

    x_correl = []
    nb_subjects = len(df_bootstrap)
    subject_label = df_bootstrap["subject_label"][:nb_subjects]

    for sub in subject_label:
        x_correl.append(data[sub])

    x_correl = np.array(x_correl)

    y_sex = df_bootstrap["Gender"][:nb_subjects]  # maybe list(df["Gender"])

    # split the sample o training/test with a 80/20 % ratio
    # and stratify sex by class, also shuffle the data

    X_train, X_test, y_train, y_test = train_test_split(
        x_correl,  # x
        y_sex,  # y
        test_size=0.2,  # 80%/20% split
        shuffle=True,  # shuffle dataset before splitting
        stratify=y_sex,  # keep distribution of sex_class consistent between train and test sets
        random_state=123,
    )  # same shuffle each time

    score = []
    model = LinearSVC(max_iter=10000)
    # score = cross_val_score(model, X_train, y_train, cv=10)
    score.append(cross_val_score(model, X_train, y_train, cv=10, n_jobs=3).mean())

    model.fit(X_train, y_train)  # fit the model/ train the model
    y_pred = model.predict(X_test)

    # calculate the model accuracy
    acc_test = model.score(X_test, y_test)
    acc_train = model.score(X_train, y_train)

    # compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    df_result = pd.DataFrame(
        data=[[tp, fn], [fp, tn]], index=["Femme", "Homme"], columns=["Femme", "Homme"]
    )
    df_result = df_result.rename_axis("actual class")
    df_result = df_result.rename_axis("predicted class", axis="columns")

    return {
        "len_X_train": len(X_train),
        "len_X_test": len(X_test),
        "score": score,
        "confusion_matrix": df_result,
        "acc_test": acc_test,
        "acc_train": acc_train,
        "model": model,
    }


if __name__ == "__main__":
    study_dir = Path(__file__).absolute().parents[3]

    df_boot = pd.read_csv(study_dir / "Final_HCP_database.csv")

    img_tpl = str(study_dir / "input/sub-{0}/sub-{0}_voxelcorrelations.nii.gz")

    atlas_dest = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
    masker = NiftiLabelsMasker(atlas_dest.maps)

    data = {}
    for sub in df_boot["subject_label"]:
        img_path = img_tpl.format(sub)
        data[sub] = img4d2vector(img_path, masker)

    iteration_number = 10000
    results = Parallel(n_jobs=-1, verbose=100)(
        delayed(run)(df_boot, data) for num in range(iteration_number)
    )

    results_file = study_dir / "results" / f"models_iteration-{iteration_number}.pkl"
    dump(results, results_file)

    
    