import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc
import copy
import csv
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler,OneSidedSelection,ClusterCentroids
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning,UndefinedMetricWarning
from math import sqrt
import json
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from joblib import Parallel, delayed

def Normalize(data):
    x = data[:,:-1]
    y = data[:,-1]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    # Z-score标准化
    #print(0 in std)
    if 0 in std:
        std[np.where(std == 0)] = 1
    return np.c_[(x - mean) / std,y]

def get_files_in_folder(folder_path):
    file_list = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_list.append(file[:-4])
        if file.endswith(".dat"):
            dat_file = os.path.join(folder_path, file)
            csv_file = os.path.join(folder_path, file.replace(".dat", ".csv"))
            os.rename(dat_file, csv_file)
    return file_list

def load_data(dataname,init_seed=0,train_ratio = 0.7):
    data = []
    if os.path.exists('./data/keel/alll/'+dataname+'.npy'):
        data = np.load('./data/keel/alll/'+dataname+'.npy')
    else:
        with open('./data/keel/alll/'+dataname+'.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in tqdm(reader):
                label = '1' if 'posi' in row[-1] else '0'
                realFeature = [inst for inst in row[:-1]]
                realFeature.append(label)
                data.append(realFeature)
            
            data = np.array(data)
            rm_col = []
            for col in range(data.shape[1]):
                flag = 0
                for char in data[:,col]:
                    if any(c.isalpha() for c in char):
                        flag = 1
                if flag:
                    rm_col.append(col)
            data = np.delete(data, rm_col, axis=1)
            data = data.astype(float)
        np.save('./data/keel/alll/'+dataname+'.npy', data)
    
    data = Normalize(data)
    n = data.shape[0]
    num_train = int(train_ratio * n)
    num_pos = sum(data[:,-1])*(1-train_ratio)
    for seed in range(99,199):
        np.random.seed(init_seed*seed)
        random.seed(init_seed*seed)
        np.random.shuffle(data)
        train_set = data[:num_train]
        test_set = data[num_train:]
        X_train = train_set[:,:-1]
        y_train = train_set[:,-1]
        X_test = test_set[:, :-1]
        y_test = test_set[:, -1]
        if sum(y_test) >= num_pos or train_ratio == 1:
            break
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int64)
    return X_train,y_train,X_test,y_test


def xyzPlot(x,PosInd,NegInd,kimsInd, z):
    z = np.array(z)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    print(PosInd,NegInd,kimsInd[0])
    print(z[PosInd])
    ax.scatter(x[PosInd,kimsInd[0]], x[PosInd,kimsInd[1]], z[PosInd], marker='o')
    ax.scatter(x[NegInd,kimsInd[0]], x[NegInd,kimsInd[1]], z[NegInd], marker='^')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def addNoise(y, inpure,mod = 'thre', thre = 0):
    if mod == 'random':
        fblack = np.array([prob if prob > thre else 0 for prob in inpure])
    else:
        fblack = np.array([1 if prob > thre else 0 for prob in inpure])
    y_n =[]
    np.random.seed(1995)
    for i, y in enumerate(y):
        if y == 0 and fblack[i] >= np.random.uniform(0,1):
            y_n.append(1) #flip label from neg to pos
        else:
            y_n.append(y)
    return np.array(y_n, dtype=np.int32)


def tune_label_noise(net_x, net_y,model = 'knn', ep = 1000):
    if model == 'knn':
        thres = np.linspace(0, 2, num=10, endpoint=True)
    else:
        thres = np.linspace(2, 4, num=10, endpoint=True)
    paras = {}
    y_ns = {}
    if net_x.shape[1] == 25:
        eps = [5,10,50,100]
    else:
        eps = [100,200,400]
    for e in eps:
        idx,(score, k, y_n) = get_fast_paras(thres, 1, tx=net_x, ty=net_y, t=e, mod = 'noise', cvmodel = model, metric = 'f1')
        paras[(k,e,thres[idx])] = score
        y_ns[(k,e,thres[idx])] = y_n
    
    for key, value in paras.items():
        if value == max(paras.values()):
            best_para = key
            break
    print(sum(net_y),sum(y_ns[best_para]))
    return net_x,y_ns[best_para],best_para

def resampling(x, y, method = 'smote', ratio = 0.5, k = 5, seed = 12):
    if method == 'smote':
        resamplor = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=k)
    elif method == 'ADASYN':
        resamplor = ADASYN(sampling_strategy=ratio, random_state=seed, n_neighbors=k)
    elif method == 'Borderline':
        resamplor = BorderlineSMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=k)
    elif method == 'Under':
        resamplor = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    elif method == 'OSS':
        resamplor = OneSidedSelection(sampling_strategy='auto', random_state=seed, n_neighbors=k, n_seeds_S=ratio)
    elif method == 'CC':
        resamplor = ClusterCentroids(sampling_strategy=ratio,random_state=42)
    #print(y,k)
    if sum(y) == 1:
        new_x, new_y = x, y
    else:

        try:
            new_x, new_y = resamplor.fit_resample(x, y)
        except:
            resamplor = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=k)
            new_x, new_y = resamplor.fit_resample(x, y)
    #print(sum(y),sum(new_y))
    return new_x, new_y

def tune_resampling(net_x, net_y, method = 'smote', model = 'knn'):
    new_X_train = []
    new_y_train = []

    n_pos = len(np.where(net_y==1)[0])
    n_neg = len(np.where(net_y==0)[0])
    if method in ['smote', 'ADASYN', 'Borderline']:
        kns = np.unique(np.linspace(1, min(20,n_pos), num=3, endpoint=False, dtype = int))
        ratios = np.linspace(n_pos/n_neg, 1, num=10, endpoint=True)[1:]
    elif method in ['CC']:
        kns = [1]
        ratios = np.linspace(n_pos/n_neg, 1, num=10, endpoint=True)[1:]
    elif method == 'OSS':
        kns = [1]
        ratios = np.linspace(2, 30, num=10, endpoint=False, dtype = int)
    else:
        kns = [1]
        ratios = np.linspace(n_pos/n_neg, 1, num=10, endpoint=True)[1:]
    paras = {}
    news = {}
    for kn in kns:
        idx,(score, k,(new_x,new_y)) = get_fast_paras(ratios, kn, net_x, net_y, mod = 'resample', method=method,cvmodel = model, metric = 'f1')
        paras[(k,kn,ratios[idx])] = score
        news[(k,kn,ratios[idx])] = (new_x, new_y)
    for key, value in paras.items():
        if value == max(paras.values()):
            best_para = key
            break
    new_X_train = news[best_para][0]
    new_y_train= news[best_para][1]
    
    return new_X_train,new_y_train,best_para

@ignore_warnings(category=(ConvergenceWarning,UndefinedMetricWarning))
def cvs(net_x, net_y_n, net_y, cvmodel = 'knn', folds = 5, metric = 'f1'):
    @ignore_warnings(category=(ConvergenceWarning,UndefinedMetricWarning))
    def process_fold(train_index, test_index, X, y_n,y, metric):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_n[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return f1 if metric == 'f1' else acc
    #model = LogisticRegression()
    #model = MLPClassifier(hidden_layer_sizes=[5], max_iter=500)
    kf = RepeatedKFold(n_splits=folds, n_repeats=3, random_state=0)
    scores = []
    if cvmodel == 'knn':
        kns = [5,10]#np.linspace(2, 30, num=10, endpoint=False, dtype = int)
        for kn in kns:
            score_k = []
            model = KNeighborsClassifier(n_neighbors=min(kn,int(len(net_y)*0.5)))
            for train_index, test_index in kf.split(net_x):
                X_train, X_test = net_x[train_index], net_x[test_index]
                y_train, y_test = net_y_n[train_index], net_y[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                if metric == 'f1':
                    score_k.append(f1) 
                else:
                    score_k.append(acc) 
            scores.append(sum(score_k) / len(score_k))
        for i,s in enumerate(scores):
            if s == max(scores):
                return s,kns[i]
    if cvmodel == 'cart':
        model = DecisionTreeClassifier()
        scores = Parallel(n_jobs=16)(delayed(process_fold)(train_index, test_index, net_x, net_y_n, net_y, metric) for train_index, test_index in kf.split(net_x))
        return sum(scores) / len(scores),0
    if cvmodel == 'mlp':
        #print('start cv')
        model = MLPClassifier(hidden_layer_sizes=[5,10,5], max_iter=800, random_state=2024)
        scores = Parallel(n_jobs=12)(delayed(process_fold)(train_index, test_index, net_x, net_y_n, net_y,metric) for train_index, test_index in kf.split(net_x))
        return sum(scores) / len(scores),0

@ignore_warnings(category=(ConvergenceWarning,UndefinedMetricWarning))
def getTestResult(X_train,y_train,X_test,y_test,k=5,testmodel = 'knn'):
    if testmodel == 'knn':
        model = KNeighborsClassifier(n_neighbors=min(k,len(y_train)))
        model.fit(X_train, y_train)
        #print(y_train,k)
        y_pred_prob = model.predict_proba(X_test)
        y_pred = [1 if prob>=0.5 else 0 for prob in y_pred_prob[:,1]]
        conf_matrix = confusion_matrix(y_test, y_pred)
    if testmodel == 'cart':
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)
        y_pred = [1 if prob>=0.5 else 0 for prob in y_pred_prob[:,1]]
        conf_matrix = confusion_matrix(y_test, y_pred)
    if testmodel == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=[5,10,5], max_iter=800, random_state=2024)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)
        y_pred = [1 if prob>=0.5 else 0 for prob in y_pred_prob[:,1]]
        conf_matrix = confusion_matrix(y_test, y_pred)
    if len(conf_matrix) == 1:
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        F1,Gmean,AUC,precision,recall = 0,0,0,0,0
    else:
        TP = conf_matrix[1, 1]
        FN = conf_matrix[1, 0]
        FP = conf_matrix[0, 1]
        TN = conf_matrix[0, 0]
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        F1 = 2*TP/(2*TP+FP + FN)
        precision = TP/(TP+FP) if (TP+FP) else 0
        recall = TP/(TP+FN)
        Gmean = np.sqrt(precision*recall) if (TP+FP) else 0
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)
        AUC = auc(fpr, tpr)
    return F1,Gmean,AUC,precision,recall,accuracy

@ignore_warnings(category=(ConvergenceWarning,UndefinedMetricWarning))
def get_flip_rate(x, y,epch=2000):
    np.random.seed(2024)
    if x.shape[1] == 25:
        proto = KNeighborsClassifier(n_neighbors = epch)
    else:
        proto = MLPClassifier(hidden_layer_sizes=[5,10,5], max_iter=epch, random_state=2024)
    PosInd = np.where(y == 1)[0]
    NegInd = np.where(y == 0)[0]
    proto.fit(x, y)
    NNs = proto.predict_proba(x)[:,1]
    preds_mean,preds_std = np.mean(NNs[NegInd]),np.std(NNs[NegInd])
    NNs[PosInd] = 0
    NNs[NegInd] = (NNs[NegInd] - preds_mean)/preds_std
    return NNs


def get_fast_paras(paras, kn, tx, ty, t=1000,mod = 'resample',method='null',cvmodel = 'knn', metric = 'f1'):
    flag = 1
    start,end = 0,len(paras)-1
    tested_dic = {}
    tested_para = set()
    if mod == 'noise':
        rhox =  get_flip_rate(tx,ty,epch = t)
    else:
        rhox = 0
    while flag:
        test_idx = set(np.linspace(start, end, num=5, endpoint=True, dtype = int))
        untested_idx = test_idx - tested_para
        if len(untested_idx) == 0:
            flag = 0
        for idx in untested_idx:
            if idx not in tested_para:
                tested_para.add(idx)
                if mod == 'resample':
                    new_x, new_y = resampling(tx,ty, k=kn, ratio=paras[idx], seed=0, method=method)
                    score, k = cvs(new_x, new_y, new_y, folds = 4, metric = metric, cvmodel = cvmodel)
                    tested_dic[idx] = (score,k,(new_x,new_y))
                else:
                    #rhox =  get_flip_rate(tx,ty, thre=paras[idx])
                    fliprate = rhox.copy()
                    fliprate[np.where(ty == 0)[0]] = np.tanh(fliprate[np.where(ty == 0)[0]]-paras[idx])
                    y_n = addNoise(ty,inpure=fliprate, mod='random', thre=0.0)
                    score, k = cvs(tx, y_n, ty, folds = 4, metric = metric, cvmodel = cvmodel)
                    tested_dic[idx] = (score,k,y_n)
        sortidx = sorted(tested_para)
        for idx, dic in tested_dic.items():
            if dic[0] == max([v[0] for v in tested_dic.values()]):
                max_idx = sortidx.index(idx)
                if idx == 0:
                    end = sortidx[max_idx+1]
                elif idx == len(paras)-1:
                    start = sortidx[max_idx-1]
                else:
                    start = sortidx[max_idx - 1]
                    end = sortidx[max_idx + 1]
                break
    for idx, dic in tested_dic.items():
        if dic[0] == max([v[0] for v in tested_dic.values()]):
            return idx,dic

def generate_simu(ntrain=500,ntest = 3000,p=0.5,m=3,d=5,seed = 0):
    n = ntrain + ntest
    posn = int(n * p)
    np.random.seed(seed)
    positive_samples = np.random.normal(0, 1, (posn, d))
    num_negative_samples = n - posn
    np.random.seed(seed)
    negative_samples = np.random.normal(0, 1, (num_negative_samples, d))
    negative_samples[:,0] = np.random.normal(m, 1, num_negative_samples)
    data = np.vstack((positive_samples, negative_samples))
    labels = np.zeros(n)
    labels[:posn] = 1 

    np.random.seed(seed)
    idx = np.random.permutation(n)
    data = data[idx]
    labels = labels[idx]
    return data[:ntrain], labels[:ntrain],data[ntrain:], labels[ntrain:]

if __name__ == '__main__':
    if os.path.exists('resultdicSimu_character_N_withErrorBar.json'):
        with open('resultdicSimu_character_N_withErrorBar.json') as j:
            resultSimu = json.loads(j.read())
    else:
        resultSimu = {}
    methodS = ['clean','noise','smote','ADASYN','Borderline','OSS','CC','Under']
    modelS = ['cart','knn']
    Ntrains = [200,400,800,1600,3200,6400]
    Ntest = 2000
    ps = [0.1]
    ms = [2]
    ds = [5]
    for Ntrain in Ntrains:
        for p in ps:
            for m in ms:
                for d in ds:
                    for smodel in modelS:
                        for method in methodS:
                            if str([p,Ntrain,d,smodel,method]) in resultSimu:
                                print('already done:',[p,Ntrain,d,smodel,method], resultSimu[str([p,Ntrain,d,smodel,method])])
                                continue
                            results = []
                            print([p,Ntrain,d,smodel,method])
                            for rounds in tqdm(range(10),total = 10):
                                X_train,y_train,X_test,y_test = generate_simu(ntrain=Ntrain,ntest = Ntest,p=p,m=m,d=d, seed = rounds)
                                if method == 'clean':
                                    score, k = cvs(X_train, y_train, y_train, folds = 4, metric = 'f1',cvmodel = smodel)
                                    F1,Gmean,AUC,precision,recall,accuracy = getTestResult(X_train,y_train,X_test,y_test,k=k,testmodel = smodel)
                                    paras = (0,0,0)
                                elif method == 'noise':
                                    noise_X, noise_y,paras = tune_label_noise(X_train, y_train,model=smodel)
                                    k = paras[0]
                                    F1,Gmean,AUC,precision,recall,accuracy = getTestResult(noise_X,noise_y,X_test,y_test,k=k,testmodel = smodel)
                                else:
                                    resample_X, resample_y,paras = tune_resampling(X_train, y_train,method = method,model=smodel)
                                    k = paras[0]
                                    F1,Gmean,AUC,precision,recall,accuracy = getTestResult(resample_X,resample_y,X_test,y_test,k=k,testmodel = smodel)
                                results.append([F1,Gmean,AUC,precision,recall,accuracy]+list(paras))
                            resultSimu[str([p,Ntrain,d,smodel,method])] = list(np.mean(results,axis=0)) + list(np.std(results,axis=0))
                            print(list(np.mean(results,axis=0))+ list(np.std(results,axis=0)))
                            with open('resultdicSimu_character_N_withErrorBar.json', 'w') as file:
                                print('data_saved')
                                json.dump(resultSimu, file)

    
    