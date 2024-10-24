import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc
import copy
import csv
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
#from cuml.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler,OneSidedSelection,ClusterCentroids
from imblearn.combine import SMOTEENN
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
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
            #print(sum(y_test),seed, 'enough pos')
            break
        #print(sum(y_test),seed, 'no enough pos')
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int64)
    #print('Train N = %d dim = %d' % X_train.shape)
    #print(round(len(y_train)/sum(y_train)))
    #print('Minority %d, Majority %d, IR = %f' % (sum(y_train),len(y_train)-sum(y_train),round(len(y_train)/sum(y_train),3)))
    #print('Test N = %d dim = %d' % X_test.shape)
    #print('Minority %d, Majority %d, IR = %f' % (sum(y_test),len(y_test)-sum(y_test),round(len(y_test)/sum(y_test),3)))
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
    print('why not show?????')

def addNoise(y, inpure,mod = 'thre', thre = 0):
    
    #xyzPlot(x,PosInd,NegInd,[1,2],inpure)
    if mod == 'random':
        fblack = np.array([prob if prob > thre else 0 for prob in inpure])
    else:
        fblack = np.array([1 if prob > thre else 0 for prob in inpure])
    #xyzPlot(x,PosInd,NegInd,kimsInd,fblack)
    y_n =[]
    np.random.seed(1995)
    for i, y in enumerate(y):
        if y == 0 and fblack[i] >= np.random.uniform(0,1):
            y_n.append(1) #flip label from neg to pos
        else:
            y_n.append(y)
    return np.array(y_n, dtype=np.int32)

def tune_label_noise(net_x, net_y,model = 'knn'):
    thres = np.linspace(0, 4, num=20, endpoint=True)
    paras = {}
    y_ns = {}
    if model == 'knn':
        eps = [200,700,1000]
        thres = np.linspace(0, 4, num=10, endpoint=True)
    else:
        eps = [1000,2000,4000]

    for e in eps:
        idx,(score, k, y_n) = get_fast_paras(thres, 1, tx=net_x, ty=net_y, t=e, mod = 'noise', cvmodel = model, metric = 'f1')
        paras[(k,e,thres[idx])] = score
        y_ns[(k,e,thres[idx])] = y_n
    
    for key, value in paras.items():
        #print(key, value)
        if value == max(paras.values()):
            best_para = key
            #print('bestPara:', best_para)
            break
    return net_x,y_ns[best_para],best_para

def resampling(x, y, method = 'smote', ratio = 0.5, k = 5, seed = 12):
    if method == 'smote':
        resamplor = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=k)
    elif method == 'ADASYN':
        resamplor = ADASYN(sampling_strategy=ratio, random_state=seed, n_neighbors=k)
        #resamplor = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=k)
    elif method == 'Borderline':
        resamplor = BorderlineSMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=k)
    elif method == 'Under':
        resamplor = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    elif method == 'OSS':
        resamplor = OneSidedSelection(sampling_strategy='auto', random_state=seed, n_neighbors=k, n_seeds_S=ratio)
    elif method == 'CC':
        #resamplor = CC(sampling_strategy=ratio, n_neighbors=k,version=1)
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
    return new_x, new_y

def tune_resampling(net_x, net_y, method = 'smote', model = 'knn'):
    new_X_train = []
    new_y_train = []

    n_pos = len(np.where(net_y==1)[0])
    n_neg = len(np.where(net_y==0)[0])
    if method in ['smote', 'ADASYN', 'Borderline']:
        kns = np.unique(np.linspace(1, min(20,n_pos), num=3, endpoint=False, dtype = int))
        ratios = np.linspace(n_pos/n_neg, 1, num=5, endpoint=True)[1:]
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
    kf = RepeatedKFold(n_splits=folds, n_repeats=2, random_state=0)
    scores = []
    if cvmodel == 'knn':
        kns = [5,10]
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
        model = MLPClassifier(hidden_layer_sizes=[5,10,5], max_iter=2000, random_state=2024)
        scores = Parallel(n_jobs=16)(delayed(process_fold)(train_index, test_index, net_x, net_y_n, net_y,metric) for train_index, test_index in kf.split(net_x))
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
        model = MLPClassifier(hidden_layer_sizes=[5,10,5], max_iter=2000, random_state=2024)
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
    proto = MLPClassifier(hidden_layer_sizes=[5,10,5], max_iter=epch, random_state=2024)
    PosInd = np.where(y == 1)[0]
    NegInd = np.where(y == 0)[0]
    proto.fit(x, y)
    NNs = proto.predict_proba(x)[:,1]
    preds_mean,preds_std = np.mean(NNs[NegInd]),np.std(NNs[NegInd])
    NNs[PosInd] = 0
    NNs[NegInd] = (NNs[NegInd] - preds_mean)/preds_std
    #NNs[NegInd] = np.tanh(NNs[NegInd]-thre)

    return NNs

def get_fast_paras(paras, kn, tx, ty, t=1000,mod = 'resample',method='null',cvmodel = 'knn', metric = 'f1'):
    flag = 1
    start,end = 0,len(paras)-1
    tested_dic = {}
    tested_para = set()
    if mod != 'resample':
        rhox =  get_flip_rate(tx,ty,epch = t)
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

if __name__ == '__main__':
    if os.path.exists('resultdicAll.json'):
        with open('resultdicAll.json') as j:
            resultdic = json.loads(j.read())
    else:
        resultdic = {}
    methodS = ['clean','noise','smote','ADASYN','Borderline','OSS','CC','Under']
    modelS = ['knn','cart','mlp']

    resultmatrix = {}
    for key, value in resultdic.items():
        key = eval(key)
        if key[0] not in resultmatrix:
            resultmatrix[key[0]] = {}
        if key[1] not in resultmatrix[key[0]]:
            resultmatrix[key[0]][key[1]] = {}
        resultmatrix[key[0]][key[1]][key[2]] = value

    fileresult_dict = {}
    for filename, models in resultmatrix.items():
        fileres = []
        for model, methods in models.items():
            allres = []
            for method, results in methods.items():
                allres.append(results[0])
            fileres.append(allres)
        fileresult_dict[filename] = np.array(fileres)

    if os.path.exists('resultdicAll_filtered.json'):
        with open('resultdicAll_filtered.json') as j:
            resultRefine = json.loads(j.read())
    else:
        resultRefine = {}

    filter_only  = 1
    cnt = 0
    for fi, res in fileresult_dict.items():
        if np.max(res[:,0]) > 0.90 or np.max(res) < 0.1 and filter_only:
            continue
        X_train,y_train,_,_ = load_data(fi,init_seed=0,train_ratio = 1)
        n,p = X_train.shape[0], X_train.shape[1]
        class_label, class_cnt = np.unique(y_train, return_counts=True)
        npos = min(class_cnt)
        nneg = max(class_cnt)
        imbratio = nneg/npos
        if npos < 30:
            continue
        print(fi,n,p,npos,nneg,imbratio)
        print(np.round(res,2))
        cnt+=1
        for smodel in modelS:
            for method in methodS:
                if str((fi,smodel,method)) in resultRefine:
                    print('already done:',fi,smodel,method, resultRefine[str((fi,smodel,method))])
                    continue
                results = []
                print(fi,smodel,method)
                for rounds in tqdm(range(10),total = 10):
                    X_train,y_train,X_test,y_test = load_data(fi,init_seed=rounds,train_ratio = 0.7)
                    if method == 'clean':
                        score, k = cvs(X_train, y_train, y_train, folds = 4, metric = 'acc',cvmodel = smodel)
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
                resultRefine[str((fi,smodel,method))] = list(np.mean(results,axis=0)) + list(np.std(results,axis=0))
                print(list(np.mean(results,axis=0)))

                with open('resultdicAll_filtered.json', 'w') as file:
                    print('data_saved')
                    json.dump(resultRefine, file)

    