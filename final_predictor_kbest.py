# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:18:06 2020

@author: IEO5115
"""

import numpy as np
import os,sys
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest,f_classif
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer,recall_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,normalize,MaxAbsScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
from sklearn.calibration import calibration_curve


def PrintFeatures(selector, out_folder, perc_threshold,feat_lab,X,y):
  selector.fit_transform(X, y)
  outfile=open(os.path.join(out_folder,"ML_features.txt"),"w")
  sort_index=np.argsort(-selector.scores_)
  cc=1
  outfile.write("#Feature\tScore\tPval\tLabel\n")
  print("Selected features: ", perc_threshold)
  for ii in sort_index:
    lab=""
    #print(cc, perc_threshold)
    if int(cc) <= int(perc_threshold):
      lab="SELECTED"
    if int(cc) > int(perc_threshold):
      lab="UNSELECTED"
    outfile.write(feat_lab[ii]+"\t"+str(selector.scores_[ii])+"\t"+str(selector.pvalues_[ii])+"\t"+lab+"\n")
    cc+=1
    
  return


def train_model(X_train,out_folder,file_h,diff_data,annova_percentile,scaler,n_features):
    X= X_train.iloc[:,5:].values
    y= X_train.true_label.values
    genes = X_train.columns[5:]

    if not n_features:
            
        svc = SVC(kernel="linear")
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),min_features_to_select=1,
                      scoring='f1')
    
    #    rfecv = RFECV(estimator=svc, step=1, cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=0),
    #                  scoring='recall')
    
        
        #rfecv.fit(X_train.iloc[:,5:].values, y_train.values)
        X_train_scaled= scaler.fit_transform(X,y)
        rfecv.fit(X_train_scaled, y)
      
        print("Optimal number of features : %d" % rfecv.n_features_)
        n_features= rfecv.n_features_
        
        #aplt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        rfecv_scores = pd.DataFrame({"features":range(1, len(rfecv.grid_scores_) + 1),
                                     "scores":rfecv.grid_scores_})
        rfecv_scores.to_csv(os.path.join(out_folder,"rfecv_scores.csv"))
        plt.savefig(os.path.join(out_folder,"optimal_features_curve.pdf"))
        plt.close()
#    transform = SelectPercentile(chi2)
#    fixed_transform = SelectKBest(chi2,56)
    
    fixed_transform = SelectKBest(f_classif,n_features)



    
    ## Pipeline with SVM
    #pipe = Pipeline([('anova', transform), ('svc', SVC(gamma='auto', class_weight="balanced", probability=True, max_iter=1000))])
#    pipe = Pipeline([('anova', transform), ('svc', SVC(gamma='auto', class_weight="balanced", probability=True))])
    pipe = Pipeline([('anova', fixed_transform),('scaler', scaler), ('svc', SVC(gamma='auto', class_weight="balanced", probability=True))])
#    pipe = Pipeline([('anova', transform),('svc', SVC(gamma='auto', class_weight="balanced", probability=True))])
#    pipe = Pipeline([('anova', transform),('scaler', MaxAbsScaler()), ('svc', SVC(gamma='auto', class_weight="balanced", probability=True))])

    ## Pipeline with Logreg
    #pipe = Pipeline([('anova', transform), ('logreg', logreg(class_weight="balanced", n_jobs=-1))])
    if annova_percentile:
        param_grid = {
            'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
            'svc__kernel' : ['linear', 'rbf', 'poly'],
            #'svc__gamma' : [0.001, 0.01, 0.1, 1],
            'anova__percentile': [annova_percentile]
        }

    elif diff_data:
        param_grid = {
            'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
            'svc__kernel' : ['linear', 'rbf', 'poly'],
            #'svc__gamma' : [0.001, 0.01, 0.1, 1],
            'anova__percentile': [100]
        }

    else:

        param_grid = {
            'svc__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
            'svc__kernel' : ['linear','rbf']
            #'svc__gamma' : [0.001, 0.01, 0.1, 1],
#            'anova__percentile': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            
        }

    '''
    param_grid = {
        'logreg__C': [100, 1000, 10000],
        'logreg__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        #'svc__gamma' : [0.001, 0.01, 0.1, 1],
        'anova__percentile': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    }
    '''

    #scoring_f='f1'
    #scoring_f='roc_auc'
    #scoring_f='accuracy'
    njobs=1
    #new_scorer = make_scorer(f1_score)
    #scoring_f=new_scorer
#    scoring_f = make_scorer(recall_score)
    scoring_f = make_scorer(f1_score)
#    scoring_f = make_scorer(precision_score)
    

    #search = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=False, scoring=scoring_f, n_jobs=njobs)
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=False, scoring=scoring_f, n_jobs=njobs)
    
    search.fit(X,y)
    
    #fixed_transform.fit_transform(X,y)
    

    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    file_h = write_line(file_h,"Best parameter (CV score=%0.3f):" % search.best_score_)
    file_h = write_line(file_h,"Best parameters\t" +str(search.best_params_))
    print(search.best_params_)

    ###Now using optimal parameters to run a repeated stratified kfold predictions, to get a best model out of many iterations
    ###Checking for various metrics, as well as for stability (STDV) and over-fitting (the Salzberg test)


#    final_params={
#        'svc__C': [search.best_params_['svc__C']],
#        'svc__kernel' : [search.best_params_['svc__kernel']],
#        'anova__percentile': [search.best_params_['anova__percentile']]
#    }


    final_params={
        'svc__C': [search.best_params_['svc__C']],
        'svc__kernel' : [search.best_params_['svc__kernel']]
        
    }


    '''
    final_params={
        'logreg__C': [search.best_params_['logreg__C']],
        'logreg__solver' : [search.best_params_['logreg__solver']],
        'anova__percentile': [search.best_params_['anova__percentile']]
    }
    '''
    ###Printing out selected features for furtger inspection
    #Outname=sys.argv[1].split("/")[-1]
    
#    print(X.shape[1], search.best_params_['anova__percentile'])
#    file_h = write_line(file_h, "Total features %s Annova Percentile %3.3f"% (X.shape[1], search.best_params_['anova__percentile']))
#    file_h = write_line(file_h, "Selected genes count: %3.3f"% ((float(X.shape[1])/100)*float(search.best_params_['anova__percentile'])))
   
 #   PrintFeatures(X,y,SelectPercentile(chi2, percentile=100), out_folder, (float(X.shape[1])/100)*float(search.best_params_['anova__percentile']),genes,X,y)
    PrintFeatures(SelectKBest(f_classif, n_features), out_folder, n_features,genes,X,y)


    ###Making a GridSearchCV run with best parameters and RepeatedStratifiedKFold search
    ###I've already tried a single GridSearchCV with paremeters selection and repeated StratifiedKfold search but it takes ways toooo long.

    num_splits = 5
    num_rep = 500
    rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_rep, random_state=0)
#    final_search = GridSearchCV(pipe, final_params, iid=False, cv=rskf, return_train_score=True, scoring=scoring_f, n_jobs=njobs)
#    final_search_rnd = GridSearchCV(pipe, final_params, iid=False, cv=rskf, return_train_score=True, scoring=scoring_f, n_jobs=njobs)

    final_search = GridSearchCV(pipe, final_params, iid=False, cv=rskf, return_train_score=True, scoring=scoring_f, n_jobs=njobs)
    final_search_rnd = GridSearchCV(pipe, final_params, iid=False, cv=rskf, return_train_score=True, scoring=scoring_f, n_jobs=njobs)


#    final_search = GridSearchCV(pipe, final_params, iid=False, cv=rskf, return_train_score=True, scoring=make_scorer(f1_score), n_jobs=njobs)
#    final_search_rnd = GridSearchCV(pipe, final_params, iid=False, cv=rskf, return_train_score=True, scoring=make_scorer(f1_score), n_jobs=njobs)


    ##Generating Random labels for Salzberg test
    #y_rnd = np.random.permutation(y)
    y_rnd = np.random.RandomState(42).permutation(y)

    ## Fit the model
    final_search.fit(X, y)
    final_search_rnd.fit(X, y_rnd)

    #print(search.best_params_)

    ###Printing some stat

    print("Final Best Model (CV score=%0.3f after %s Cycles):" % (final_search.best_score_, num_rep))
    print("Random Salzberg (CV score=%0.3f after %s Cycles):" % (final_search_rnd.best_score_, num_rep))

    
    file_h = write_line(file_h, str("Final Best Model (CV score=%0.3f after %s Cycles):" % (final_search.best_score_, num_rep)))
    file_h = write_line(file_h, str("Random Salzberg (CV score=%0.3f after %s Cycles):" % (final_search_rnd.best_score_, num_rep)))



    ### Here I calculate the mean of test sets in every iteration (i.e. one mean value per iteration)
    ### and then I calculate the mean and STD of these aforementioned means
    print('-----------------------------------------------------')
    file_h =write_line(file_h, '-----------------------------------------------------')

    mean_total = []
    for r in range(0, num_rep):
        mean_iter = 0
        for s in range(r*num_splits, num_splits*(r+1)):
            mean_iter += final_search.cv_results_['split'+str(s)+'_test_score']
            #print str(s),
        #print mean_iter/float(num_splits)
        mean_total.append(mean_iter/float(num_splits))

    print("Mean score (iter-wise)="+str(round(np.mean(mean_total), 2)))
    print("Mean STDV (iter-wise)="+str(round(np.std(mean_total), 2)))
    print('-----------------------------------------------------')
    file_h =write_line(file_h, '-----------------------------------------------------')

    file_h = write_line(file_h ,str("Mean score (iter-wise)="+str(round(np.mean(mean_total), 2))))
    file_h =write_line(file_h, str("Mean STDV (iter-wise)="+str(round(np.std(mean_total), 2))))
    file_h =write_line(file_h, '-----------------------------------------------------')


    ### Here is the mean and STD of all the test scores
    print("Mean score (overall)="+str(final_search.cv_results_['mean_test_score'][0]))
    print("Mean STDV (overall)="+str(final_search.cv_results_['std_test_score'][0]))

    file_h = write_line(file_h, str("Mean score (overall)="+str(final_search.cv_results_['mean_test_score'][0])))
    file_h = write_line(file_h, str("Mean STDV (overall)="+str(final_search.cv_results_['std_test_score'][0])))


    return (final_search,file_h)


def rescore(y_pred,t):
    if not t:
        t=0.5
    new_fixed_y = [1 if y >= t else 0 for y in y_pred]
    return new_fixed_y

    

def test_model(final_search,X_test_df,out_folder,file_h,scaler,threshold,display_labels):
    fig, ax = plt.subplots()

    X_test = X_test_df.iloc[:,5:].values
    y_test = X_test_df.true_label.values
    #X_test_scaled = scaler.fit_transform(X_test)
    y_pred=final_search.predict_proba(X_test)[:,1]
    y_pred_default=final_search.predict(X_test)
    y_pred_decision=final_search.decision_function(X_test)
    
    y_pred_fixed = rescore(y_pred,threshold)
    #y_pred_fixed=final_search.predict(X_test)
    y_pred_log=final_search.predict_log_proba(X_test)[:,1]

    average_precision = average_precision_score(y_test, y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    file_h=write_line(file_h,'Average precision-recall score: {0:0.2f}'.format(average_precision))
    
    ###Plotting results
    log_fpr, log_tpr, log_threshold = roc_curve(y_test, y_pred)
#    sample_names = [label_dict[val] for val in y_test_labels]
    #sample_names = [label_dict[val] for val in X_test_df.index]
    #print(sample_names)
    
    
    pred_result = X_test_df.iloc[:,:5]

    this_result_default = pd.Series(y_pred_default,index=pred_result.index)
    this_result_default.name="predicted_label"

    this_result_dec = pd.Series(y_pred_decision,index=pred_result.index)
    this_result_dec.name="decision_function"



    this_result = pd.Series(y_pred,index=pred_result.index)
    this_result.name="prediction_proba"

    this_result_fixed = pd.Series(y_pred_fixed,index=pred_result.index)
    this_result_fixed.name="predicted_label_rescored"


    this_result_log = pd.Series(y_pred_log,index=pred_result.index)
    this_result_log.name="prediction_log_proba"

    #zipped = zip(y_test_labels,y_test,y_pred,sample_names)
    pd.concat([pred_result,this_result,this_result_log,this_result_default,this_result_fixed,this_result_dec],axis=1).to_csv(os.path.join(out_folder,"pred_result.csv"))

    file_h =write_line(file_h, '-----------------------------------------------------')
    file_h =write_line(file_h, 'Prediction Result...')
    
#    file_h =write_line(file_h, '\tSample\tTest_Label\tPredicted_Label\tSample_names\t')
    
    #for val in zipped:
    #    file_h = write_line(file_h, "\t%s\t%s\t%0.3f\t%s\t" %(val[0],val[1],val[2],val[3]))
    log_roc_auc = auc(log_fpr, log_tpr)
    file_h =write_line(file_h, '-----------------------------------------------------')
    file_h =write_line(file_h, 'Test Results.......')
    file_h =write_line(file_h, 'Log_Thres\tLog_TPR\tLog_FPR\tLog_TPR-Log_FPR')
    for ii in range(len(log_tpr)):
        print(log_threshold[ii], log_tpr[ii], log_fpr[ii], log_tpr[ii]-log_fpr[ii])
        file_h =write_line(file_h, "%0.5f\t%0.5f\t%0.5f\t%0.5f" %(log_threshold[ii], log_tpr[ii], log_fpr[ii], log_tpr[ii]-log_fpr[ii]))

    
    plt.plot(log_fpr, log_tpr, color='orangered', linestyle='--', label='ROC curve (area = %0.3f)' % log_roc_auc, lw=3)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')

    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Performance on validation set (%s)' % y_test.shape[0])
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_folder,"roc_curve.pdf"))

    #label_dict = expdesign.set_index("condition_rep")["label"].to_dict()
    plt.close()   
    
    disp = plot_precision_recall_curve(final_search, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
    plt.savefig(os.path.join(out_folder,"precision_recall_curve.pdf"))
    plt.close()
    
    cm = confusion_matrix(y_test, y_pred_fixed)
    tn, fp, fn, tp =cm.ravel()
    print("tn\tfp\tfn\ttp")
    print(tn,"\t",fp,"\t",fn,"\t",tp)
    
    file_h = write_line(file_h,"---------Confusion_matrix-------")

    file_h = write_line(file_h,"tn\tfp\tfn\ttp")

    file_h =write_line(file_h,str(tn)+"\t"+str(fp)+"\t"+str(fn)+"\t"+str(tp))
    file_h = write_line(file_h,"----------------")
#    display_labels=["Healthy","Tumor"]
#    display_labels=["No Relapse","Relapse"]
    display_labels=display_labels

    
    
    disp = plot_confusion_matrix(final_search, X_test, y_test,
                                 display_labels=display_labels,
                                 cmap=plt.cm.Blues)

    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(os.path.join(out_folder,"default_confusion_matrix.pdf"))
    plt.close()


    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp.plot(cmap=plt.cm.Reds)

    disp.ax_.set_title("Confusion Matrix")
    plt.savefig(os.path.join(out_folder,"rescored_confusion_matrix.pdf"))
    plt.close()
    print(classification_report(y_test, y_pred_fixed, target_names=display_labels))
    file_h = write_line(file_h,"----Precision Recall F1-score Support------")

    file_h = write_line(file_h,classification_report(y_test, y_pred_fixed, target_names=display_labels))
    file_h = write_line(file_h,"----------------")
    
    return file_h
        

def add_sample_labels(proteinGroups_file,expdesign_file):
    data = pd.read_csv(proteinGroups_file,sep="\t")
    expdesign = pd.read_csv(expdesign_file,sep="\t")
    expdesign["sample_name"] = expdesign.condition.str.cat(expdesign.replicate.apply(str),sep="_")
    expdesign.set_index("sample_name",inplace=True)
    data.set_index("Gene",inplace=True)
    data  = data.transpose()
    #data = data/data.max()
    #expdesign["sample_name"] = expdesign.index
    data = pd.concat([expdesign,data],axis=1)
    #data.to_csv("ML_input_data.txt",sep="\t")
    return data


def add_sample_labels_fresh(proteinGroups_file,expdesign_file):
    data = pd.read_csv("proteinGroups.txt",sep="\t")
    data = data[data["Reverse"].isnull()]
    data = data[data["Potential contaminant"].isnull()]
    data["feature"] = data["Gene names"]
    data.feature[data.feature.isnull()]= data["Protein IDs"][data.feature.isnull()]
    data.set_index("feature",inplace=True)
    data = data.filter(regex="LFQ intensity (.*)")
    
    expdesign = pd.read_csv(expdesign_file,sep="\t")
    expdesign["sample_name"] = expdesign.condition.str.cat(expdesign.replicate.apply(str),sep="_")
    expdesign["lfq_cols"] = "LFQ intensity "+expdesign.label

    data = data[expdesign.lfq_cols]
    
    expdesign.set_index("lfq_cols",inplace=True)
    
    data  = data.transpose()
    #data = data/data.max()
    #expdesign["sample_name"] = expdesign.index
    data = pd.concat([expdesign,data],axis=1)
    data.set_index("sample_name",inplace=True)
    
    
    #data.to_csv("ML_input_data.txt",sep="\t")
    return data




def create_new_file(outdir,testsize,counts,subgroup,random_state):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if counts:
        curdir = os.path.join(outdir,"count_per_group_"+str(counts))
    elif subgroup:
        curdir = os.path.join(outdir,"subgroup_random_state_"+str(random_state))
    else:
        curdir = os.path.join(outdir,"testsize_"+str(testsize))

    if os.path.exists(curdir):
        for i in range(100):
            new_curdir= os.path.join(curdir,"_"+str(i))
            #curdir=os.path.join(curdir,"_"+str(i))
            if os.path.exists(new_curdir):
                continue
            else:
                curdir= new_curdir
                break
        #sys.exit("Output folder %s already exists..Exiting!" % curdir)
    os.makedirs(curdir)
    print("Output directory is:",curdir)
    filepath = os.path.join(curdir,"ML_results.txt")
    file_h = open(filepath,"w")
    return (file_h,curdir)

def write_line(file_h,line):
    file_h.write(line+"\n")
    return file_h

def split_train_test(data, testsize,random_state):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=random_state)
    genes = data.columns.values[:-1]
    return (X_train, X_test, y_train, y_test,genes)


def split_train_test_stratify(data, testsize,random_state):
    X_train, X_test, y_train, y_test = train_test_split(data, data.true_label, test_size=testsize, stratify=data.subgroup,random_state=random_state)
    return (X_train, X_test, y_train, y_test)


def split_by_num(data,count_per_type,random_state):
    healthy_train_data = data[data.Sample_type==0].sample(count_per_type,random_state=random_state)
    #healthy_train_data = take_samples_from_subgroups(healthy_train_data,expdesign_file_path)
    tumor_train_data = data[data.Sample_type==1].sample(count_per_type,random_state=random_state)
    train_data = pd.concat([healthy_train_data,tumor_train_data])
    test_data = data.loc[~data.index.isin(train_data.index),:]

    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]

    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    genes = data.columns.values[:-1]
    return (X_train, X_test, y_train, y_test,genes)

def take_samples_from_subgroups(data,expdesign,random_state):
    asthma_labels= expdesign.condition_rep[expdesign.label.str.startswith("Asthma")].sample(3,random_state=random_state)
    bpco_labels= expdesign.condition_rep[expdesign.label.str.startswith("BPCO")].sample(8,random_state=random_state)
    healthy_labels= expdesign.condition_rep[expdesign.label.str.startswith("Healthy")].sample(7,random_state=random_state)
    obesity_labels= expdesign.condition_rep[expdesign.label.str.startswith("Obesity")].sample(4,random_state=random_state)
    tumor_labels = expdesign.condition_rep[expdesign.label.str.startswith("Tumor")].sample(22,random_state=random_state)
    train_labels = pd.concat([asthma_labels,bpco_labels,healthy_labels,obesity_labels,tumor_labels],ignore_index=True)

    train_data = data.loc[train_labels,:]

    test_data = data.loc[~data.index.isin(train_data.index),:]

    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1]

    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    genes = data.columns.values[:-1]
    return (X_train, X_test, y_train, y_test,genes)

def take_samples_from_subgroups_new(data,random_state):
    print(data.index)
    asthma_labels= data.subgroup[data.subgroup.str.startswith("Asthma")].sample(3,random_state=random_state).index.to_list()
    
    bpco_labels= data.subgroup[data.subgroup.str.startswith("BPCO")].sample(8,random_state=random_state).index.to_list()

    healthy_labels= data.subgroup[data.subgroup.str.startswith("Healthy")].sample(7,random_state=random_state).index.to_list()

    obesity_labels= data.subgroup[data.subgroup.str.startswith("Obesity")].sample(4,random_state=random_state).index.to_list()
    tumor_labels= data.subgroup[data.subgroup.str.startswith("Tumor")].sample(22,random_state=random_state).index.to_list()

#    bpco_labels= expdesign.condition_rep[expdesign.label.str.startswith("BPCO")].sample(8,random_state=random_state)
#    healthy_labels= expdesign.condition_rep[expdesign.label.str.startswith("Healthy")].sample(7,random_state=random_state)
#    obesity_labels= expdesign.condition_rep[expdesign.label.str.startswith("Obesity")].sample(4,random_state=random_state)
#    tumor_labels = expdesign.condition_rep[expdesign.label.str.startswith("Tumor")].sample(22,random_state=random_state)
 #   train_labels = pd.concat([asthma_labels,bpco_labels,healthy_labels,obesity_labels,tumor_labels],ignore_index=True)
    train_labels = asthma_labels+bpco_labels+healthy_labels+obesity_labels+tumor_labels

    train_data = data.loc[train_labels,:]

    test_data = data.loc[~data.index.isin(train_data.index),:]
    
    return (train_data,test_data,train_data.true_label,test_data.true_label)

def take_samples_from_subgroups_new_33(data,random_state):
    print(data.index)
    asthma_labels= data.subgroup[data.subgroup.str.startswith("Asthma")].sample(4,random_state=random_state).index.to_list()
    
    bpco_labels= data.subgroup[data.subgroup.str.startswith("BPCO")].sample(10,random_state=random_state).index.to_list()

    healthy_labels= data.subgroup[data.subgroup.str.startswith("Healthy")].sample(8,random_state=random_state).index.to_list()

    obesity_labels= data.subgroup[data.subgroup.str.startswith("Obesity")].sample(5,random_state=random_state).index.to_list()
    tumor_labels= data.subgroup[data.subgroup.str.startswith("Tumor")].sample(27,random_state=random_state).index.to_list()

#    bpco_labels= expdesign.condition_rep[expdesign.label.str.startswith("BPCO")].sample(8,random_state=random_state)
#    healthy_labels= expdesign.condition_rep[expdesign.label.str.startswith("Healthy")].sample(7,random_state=random_state)
#    obesity_labels= expdesign.condition_rep[expdesign.label.str.startswith("Obesity")].sample(4,random_state=random_state)
#    tumor_labels = expdesign.condition_rep[expdesign.label.str.startswith("Tumor")].sample(22,random_state=random_state)
 #   train_labels = pd.concat([asthma_labels,bpco_labels,healthy_labels,obesity_labels,tumor_labels],ignore_index=True)
    train_labels = asthma_labels+bpco_labels+healthy_labels+obesity_labels+tumor_labels

    train_data = data.loc[train_labels,:]

    test_data = data.loc[~data.index.isin(train_data.index),:]
    
    return (train_data,test_data,train_data.true_label,test_data.true_label)




def take_samples_from_subgroups_new_all(data,random_state):
    print(data.index)
    asthma_labels= data.subgroup[data.subgroup.str.startswith("Asthma")].sample(4,random_state=random_state).index.to_list()
    
    bpco_labels= data.subgroup[data.subgroup.str.startswith("BPCO")].sample(8,random_state=random_state).index.to_list()

    healthy_labels= data.subgroup[data.subgroup.str.startswith("Healthy")].sample(8,random_state=random_state).index.to_list()

    obesity_labels= data.subgroup[data.subgroup.str.startswith("Obesity")].sample(5,random_state=random_state).index.to_list()
    tumor_labels= data.subgroup[data.subgroup.str.startswith("Tumor")].sample(25,random_state=random_state).index.to_list()

#    bpco_labels= expdesign.condition_rep[expdesign.label.str.startswith("BPCO")].sample(8,random_state=random_state)
#    healthy_labels= expdesign.condition_rep[expdesign.label.str.startswith("Healthy")].sample(7,random_state=random_state)
#    obesity_labels= expdesign.condition_rep[expdesign.label.str.startswith("Obesity")].sample(4,random_state=random_state)
#    tumor_labels = expdesign.condition_rep[expdesign.label.str.startswith("Tumor")].sample(22,random_state=random_state)
 #   train_labels = pd.concat([asthma_labels,bpco_labels,healthy_labels,obesity_labels,tumor_labels],ignore_index=True)
    train_labels = asthma_labels+bpco_labels+healthy_labels+obesity_labels+tumor_labels

    train_data = data.loc[train_labels,:]

    test_data = data.loc[~data.index.isin(train_data.index),:]
    
    return (train_data,test_data,train_data.true_label,test_data.true_label)



def read_diff_file(data,diff_file_path):
    diff = pd.read_csv(diff_file_path,sep="\t")
    diff_index = diff[diff.Tumor_vs_Healthy_significant].name
    diff_data = data.loc[:,data.columns.isin(diff_index)]
    diff_data["Sample_type"] = data.Sample_type
    return diff_data

def get_expdesign(expdesign_file_path):
    expdesign = pd.read_csv(expdesign_file_path,sep="\t")
    expdesign["condition_rep"] = expdesign.condition.str.cat(expdesign.replicate.apply(str),sep="_")
    label_dict = expdesign.set_index("condition_rep")["label"].to_dict()
    subgroup_dict = expdesign.set_index("condition_rep")["subgroup"].to_dict()
    return (label_dict,subgroup_dict)


def parse_testsize(args):

    if args.testsize:
        if args.testsize.startswith("0."):
            args.testsize=float(args.testsize)
        else:
            if int(args.testsize):
                args.testsize=int(args.testsize)
            else:
                print("Using the default test size of 0.33...If this is not intended...Stop ANALYSIS")
                args.testsize=0.33
        
    return args


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-p",dest="protein_groups",
            help ="full path to protein groups file...",type=str,
            default=None)
    parser.add_argument("-e",dest="expdesign",
            help="full path to expdesign file...",type=str,default=None)
    parser.add_argument("-o",dest="out",
            help="full path to output folder...",type=str,required=True)
    parser.add_argument("-t",dest="testsize",
            help="proportion of samples as test set ...",default=None,type=str)

    parser.add_argument("-c",dest="counts",
            help="Number of sample in single group for test ...",default=None,type=int)

    parser.add_argument("-s",dest="subgroup",
            help="Take samples from sub-group for train test ...",default=False,type=bool)

    parser.add_argument("-r",dest="random",
            help="Random seed integer ...",default=42,type=int)

    parser.add_argument("-d",dest="diff_data",
            help ="full path to output file from DEP analyis with list of significant genes...",type=str,
            default=None)

    parser.add_argument("-annovaP",dest="annova_percentile",
            help ="Annova percentile cutoff for number of genes to...",type=float,
            default=None)

    parser.add_argument("-stratify",dest="stratify",
            help ="Stratify samples by subgroups (will make use of expdesign file here which should contain a subgroup column).....",type=bool,
            default=False)


    args = parser.parse_args()
    args = parse_testsize(args)
    return args


def set_arguments():
    parser = ArgumentParser()
    args = parser.parse_args()
#    args.protein_groups = "C:/projects/serum/new_analysis/final_IEO/ML/all_samples/results/LFQ_intensity.txt"
#    args.expdesign = "C:/projects/serum/new_analysis/final_IEO/ML/all_samples/93_samples.txt"
#    args.out = "C:/projects/serum/new_analysis/final_IEO/ML/all_samples/results/ML/testsize_0.1/features_20"


# =============================================================================
#    args.protein_groups = "C:/projects/serum/new_analysis/final_IEO/ML/LFQ_intensity.txt"
#    args.expdesign = "C:/projects/serum/new_analysis/final_IEO/ML/expdesign_for_IEO_RT_5_samples_removed_H192.txt"
#    args.out = "C:/projects/serum/new_analysis/final_IEO/ML/grant_0410/final/test_size_33/subgroup_33/random_state_1234/"
# 
# =============================================================================
    print(os.getcwd())
    args.protein_groups = os.path.join(os.getcwd(),"data","LFQ_intensity.txt")
    args.expdesign = os.path.join(os.getcwd(),"data","expdesign_for_IEO_RT_5_samples_removed_H192.txt")
    args.out = os.path.join(os.getcwd(),"output")
    
    #sys.exit()
    args.stratify=None
    args.random=1234
    args.testsize=33 ### change testsize along with the function
    args.annova_percentile=None
    args.counts=None
    args.subgroup=1
    args.diff_data=None
    #args.scaler=MaxAbsScaler()
    args.scaler=StandardScaler()
    args.n_features=None
    args.threshold=None
    args.display_labels=["Healthy","Tumor"]
    
    
    
    return args
    
  

def main():
    args = set_arguments()
    #print(args)
    file_h,curdir = create_new_file(args.out,args.testsize,args.counts,args.subgroup,args.random)
    file_h = write_line(file_h,"####################classifier output###########")
    file_h = write_line(file_h,"current output dir: %s"% curdir)
    for arg in args.__dict__:
        file_h = write_line(file_h, " argument_dest: %s\targument_value: %s" % (arg,args.__dict__[arg]))
    data = add_sample_labels(args.protein_groups,args.expdesign)
    data.to_csv(os.path.join(curdir,"ML_input.txt"),sep="\t")

    if args.diff_data:
        print("Getting Diff...")
        data = read_diff_file(data,args.diff_data)
    if args.counts:
        print("Getting counts....")
        X_train, X_test, y_train, y_test,genes = split_by_num(data,args.counts,args.random)
    elif args.subgroup:
        print("Hard coded subgroup...")
      
#        X_train, X_test, y_train, y_test =take_samples_from_subgroups_new(data, args.random)
#        X_train, X_test, y_train, y_test =take_samples_from_subgroups_new_all(data, args.random)
        X_train, X_test, y_train, y_test =take_samples_from_subgroups_new_33(data, args.random)

    elif args.stratify:
        print("Stratifying samples by subgroups.....")
        X_train, X_test, y_train, y_test = split_train_test_stratify(data,args.testsize,args.random)

    else:
        print("Default.....")
        X_train, X_test, y_train, y_test,genes = split_train_test(data,args.testsize,args.random)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.testsize, random_state=42)
    
    data.to_csv(os.path.join(curdir,"ML_input_data.txt"),sep="\t")
    X_train.to_csv(os.path.join(curdir,"ML_train_data.txt"),sep="\t")
    X_test.to_csv(os.path.join(curdir,"ML_test_data.txt"),sep="\t")
    file_h = write_line(file_h,"-------------------")
    file_h = write_line(file_h,"training shape :"+str(X_train.shape))
    file_h = write_line(file_h,"test shape :"+str(X_test.shape))
    model,file_h = train_model(X_train,curdir,file_h,args.diff_data,args.annova_percentile,args.scaler,args.n_features)
    file_h = test_model(model,X_test,curdir,file_h,args.scaler,args.threshold,args.display_labels)
    file_h = write_line(file_h, "-------------DONE-------------")
    file_h.close()

if __name__=="__main__":
    main()


