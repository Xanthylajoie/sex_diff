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

def machine_learning(x, y):
    # split the sample o training/test with a 80/20 % ratio 
    # and stratify sex by class, also shuffle the data
    X_train, X_test, y_train, y_test = train_test_split( 
                                                        x,  # x 
                                                        y,  # y 
                                                        test_size = 0.2, # 80%/20% split 
                                                        shuffle = True,  #shuffle dataset before splitting
                                                        stratify = y,  # keep distribution of sex_class consistent between train and test sets
                                                        random_state = 123) #same shuffle each time 

    print('train:', len(X_train),'test:', len(X_test))
    
    
    score = []
    model = LinearSVC()
    score.append(cross_val_score(model, X_train, y_train, cv=10, n_jobs = 3))
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
    
    return {
        "len_X_train": len(X_train),
        "len_X_test": len(X_test),
        "score": score,
        "confusion_matrix": df_result,
        "acc_test": acc_test,
        "acc_train": acc_train,
        "model": model,
    }


def run(df_boot, data, seed):
    df_bootstrap = pd.DataFrame()
    for j in range(0, len(df_boot)):
        index = random.randint(0, len(df_boot)-1)
        frames = [df_bootstrap, df_boot[index:index+1]]
        df_bootstrap = pd.concat(frames)

    df_bootstrap = df_bootstrap.drop(df_bootstrap.columns[0], axis=1)


    # print(df_bootstrap["Gender"])  # should have the whole list

    x_correl = []
    nb_subjects = len(df_bootstrap)
    subject_label = df_bootstrap["subject_label"][:nb_subjects]

    x_correl = []
    for sub in tqdm(subject_label):
        for seed_name in seed:
            x_correl.append(data[sub, seed_name])

    x_correl = np.array(x_correl)
    x_correl = x_correl.reshape(len(df_boot), len(seed)*148)  # autant de lignes que de sujets, autant de colones (nb region atlas x 8 (seeds))


    y_sex = df_bootstrap["Gender"][:nb_subjects]  # maybe list(df["Gender"])

    print(sum(y_sex), len(y_sex))  #double check 

    coef = machine_learning(x_correl, y_sex)

    return coef


if __name__ == "__main__":
    bootstrap_coef = []
    SEEDS = ["opIFG_L", "planumtemp_L", "aMTG_L", "pITG_L", "opIFG_R", "planumtemp_R", "aMTG_R", "pITG_R"]
    img_tpl = "/data/brambati/dataset/HCP/derivatives/seed-to-voxel-nilearn/results_3D/{seed_name}/sub-{participant_id}_{seed_name}_voxelcorrelations.nii.gz"
    atlas_dest = datasets.fetch_atlas_destrieux_2009()
    masker = NiftiLabelsMasker(atlas_dest.maps)
    
    data= {}
    
    for seed in SEEDS:
        for sub in tqdm(df_boot["subject_label"]):
            img_path = img_tpl.format(seed_name=seed, participant_id=sub)
            data[(sub, seed)] = img4d2vector(img_path, masker)
    
    # une seed a la fois
    iteration_number = 10000
    results = Parallel(n_jobs=-1, verbose=100)(
        delayed(run)(df_boot, data, seed) for seed in SEEDS for num in range(iteration_number)
    )
    
    # les seeds a gauche
    iteration_number = 10000
    results = Parallel(n_jobs=-1, verbose=100)(
        delayed(run)(df_boot, data, ["opIFG_L", "planumtemp_L", "aMTG_L", "pITG_L"]) for num in range(iteration_number)
    )
    
    # les seeds a droite
    iteration_number = 10000
    results = Parallel(n_jobs=-1, verbose=100)(
        delayed(run)(df_boot, data, ["opIFG_R", "planumtemp_R", "aMTG_R", "pITG_R"]) for num in range(iteration_number)
    )