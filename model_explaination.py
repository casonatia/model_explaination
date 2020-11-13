import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
import statsmodels.api as sm
import statsmodels.stats.api as sms
from tqdm import tqdm
import shap
import logging as log




def ICE(df, predict_fn, predict_fn_args = None, predictors_numerical = None, predictors_categorical = None,
        sample = 10000, plot_sample = None, cat_size = None, bins_size = 10, prepare_df_fn = None,
        prepare_df_fn_args = None, targets = None, multioutput = False, plot_type = 'All', savedir = None, pdf = None) :
    '''
    Implementation of Individual Conditional Expectation

    - df : Dataframe di test
    - predict_fn : funzione di predict, esempio : model.predict
    - predict_fn_args : dizionario contenente argomenti extra della funzione di predict
    - predictors_numerical : lista features numeriche 
    - predictors_categorical : lista feature categoriche
    - sample : dimensione campione dataset utilizzato per il calcolo 
    - plot_sample : linee utilizzate per il plot (se nullo == sample) 
    - cat_size : numero di categorie per le features categoriche (se nullo tutte)
    - bins_size : numero di bins per le numeriche
    - prepare_df_fn : funzione da passare al df prima del predict (ad esempio associazione layer colonne)
    - prepare_df_fn_args : argomenti extra di prepare_df_fn
    - targets : lista con i nomi dei target (solo modello con target non binario)
    - multioutput : da utilizzare se gli output sono molteplici. Es : rete neurale con un layer per classe
    - plot_type : se 'single' viene fatto un grafico per feature, se 'All' uno per tutte
    - savedir : cartella per salvare il plot
    - pdf : oggetto pdf 
    '''
    if prepare_df_fn :
        if prepare_df_fn_args :
            preprocessing = partial(prepare_df_fn, **prepare_df_fn_args)
        else : 
            preprocessing = partial(prepare_df_fn)
    if predict_fn_args :
        predict = partial(predict_fn, **predict_fn_args)
    else :
        predict = partial(predict_fn)
    if sample :
        if df.shape[0] > sample :
            d = df.dropna().sample(sample, random_state = 28)
        else :
            d = df.dropna().copy()
    else :
        d = df.dropna().copy()
    y_hat = {}
    if predictors_categorical and predictors_numerical :
        predictors =  predictors_categorical + predictors_numerical
    elif predictors_categorical and not predictors_numerical :
        predictors =  predictors_categorical
    elif not predictors_categorical and predictors_numerical :
        predictors =  predictors_numerical
    with tqdm(total = len(predictors)) as f :
        bins = [int(x) for x in np.linspace(0, 100, bins_size)]
        if predictors_numerical :
            for pred in predictors_numerical :
                percentiles = np.percentile(d[pred].values, bins)
                y_hat[pred] = {}
                for perc in percentiles :
                    d_ = d.copy()
                    d_[pred] = perc
                    if prepare_df_fn :
                        y_hat[pred][perc] = predict(preprocessing(d_))
                    else :
                        y_hat[pred][np.round(perc, 5)] = predict(d_)
                f.update()
        if predictors_categorical :
            for pred in predictors_categorical :
                y_hat[pred] = {}
                categories = np.unique(d[pred].values)
                if cat_size :
                    if len(categories) > cat_size :
                        categories = categories[0:cat_size]
                for cat in categories :
                    d_ = d.copy()
                    d_[pred] = cat
                    if prepare_df_fn :
                        y_hat[pred][cat] = predict(preprocessing(d_))
                    else :
                        y_hat[pred][cat] = predict(d_)
                f.update()
    if targets :
        if multioutput :
            for t in range(0, len(targets)) :
                if plot_type == 'single' :
                    for key in list(y_hat.keys()) :
                        matrix = np.concatenate([y_hat[key][k][t] for k in y_hat[key].keys()], axis = 1 )
                        for idx in range(0, matrix.shape[0]) :
                            matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                        plt.figure(figsize = (16,9))
                        x = list(y_hat[key].keys())
                        if plot_sample :
                            for idx in range(0, plot_sample) :
                                plt.plot(x, matrix[idx, :], c = 'black', alpha = .1)
                        else :
                            for idx in range(0, matrix.shape[0]) :
                                plt.plot(x, matrix[idx, :], c = 'black', alpha = .1)
                        avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                        plt.plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                        plt.title('{}_{}'.format(targets[t], key))
                        plt.tight_layout()
                        if savedir :
                            savename = 'ICE_{}_{}'.format(targets[t], key)
                            name = os.path.join(savedir,savename)
                            plt.savefig(name, bbox_inches = "tight")
                        if pdf :
                            pdf.savefig(bbox_inches = "tight")
                if plot_type == 'All' :
                    fig, axs = plt.subplots(int(np.ceil(len(list(y_hat.keys())) / 2)),2)
                    height = 3 * int(np.ceil(len(list(y_hat.keys())) / 2))
                    fig.set_figheight(height)
                    fig.set_figwidth(10)
                    if int(np.ceil(len(list(y_hat.keys())) / 2)) == 1 : 
                        for y_plot in range(0, len(list(y_hat.keys()))) :
                            pred = list(y_hat.keys())[y_plot]
                            matrix = np.concatenate([y_hat[pred][k][t] for k in y_hat[pred].keys()], axis = 1 )
                            for idx in range(0, matrix.shape[0]) :
                                matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                            x = list(y_hat[pred].keys())
                            if plot_sample :
                                for idx in range(0, plot_sample) :
                                    axs[y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                            else :                               
                                for idx in range(0, matrix.shape[0]) :
                                    axs[y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                            avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                            axs[y_plot].plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                            axs[y_plot].set_title(pred)
                        plt.suptitle('ICE {}'.format(targets[t]))
                        plt.subplots_adjust(top=0.85)
                        plt.tight_layout()
                        if savedir :
                            i = 0
                            suffix = ''
                            savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            while os.path.exists(os.path.join(savedir, savename)) is True :
                                i = i + 1
                                suffix = i
                                savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            name = os.path.join(savedir,savename)
                            plt.savefig(name, bbox_inches = "tight")
                        if pdf :
                            pdf.savefig(bbox_inches = "tight")
                    else :
                        i = 0
                        for x_plot in range(0, int(np.ceil(len(list(y_hat.keys())) / 2))) :
                            for y_plot in range(0, 2) :
                                pred = list(y_hat.keys())[i]
                                matrix = np.concatenate([y_hat[pred][k][t] for k in y_hat[pred].keys()], axis = 1 )
                                for idx in range(0, matrix.shape[0]) :
                                    matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                                x = list(y_hat[pred].keys())
                                if plot_sample :
                                    for idx in range(0, plot_sample) :
                                        axs[x_plot,y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                                else :
                                    for idx in range(0, matrix.shape[0]) :
                                        axs[x_plot,y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                                avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                                axs[x_plot,y_plot].plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                                axs[x_plot,y_plot].set_title(pred)
                                i = i + 1 
                                if i == len(list(y_hat.keys())) :
                                    break
                        plt.suptitle('ICE {}'.format(targets[t]))
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
                        if savedir :
                            i = 0
                            suffix = ''
                            savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            while os.path.exists(os.path.join(savedir, savename)) is True :
                                i = i + 1
                                suffix = i
                                savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            name = os.path.join(savedir,savename)
                            plt.savefig(name, bbox_inches = "tight")
                        if pdf :
                            pdf.savefig(bbox_inches = "tight")
        else :
            for t in range(0, len(targets)) :
                if plot_type == 'single' :
                    for key in list(y_hat.keys()) :
                        matrix = np.concatenate([y_hat[key][k][:,t] for k in y_hat[key].keys()], axis = 1 )
                        for idx in range(0, matrix.shape[0]) :
                            matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                        plt.figure(figsize = (16,9))
                        x = list(y_hat[key].keys())
                        if plot_sample :
                             for idx in range(0, plot_sample) :
                                plt.plot(x, matrix[idx, :], c = 'black', alpha = .1)
                        else :
                            for idx in range(0, matrix.shape[0]) :
                                plt.plot(x, matrix[idx, :], c = 'black', alpha = .1)
                        avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                        plt.plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                        plt.title('{}_{}'.format(targets[t], key))
                        plt.tight_layout()
                        if savedir :
                            savename = 'ICE_{}_{}'.format(targets[t], key)
                            name = os.path.join(savedir,savename)
                            plt.savefig(name, bbox_inches = "tight")
                        if pdf :
                            pdf.savefig(bbox_inches = "tight")
                if plot_type == 'All' :
                    fig, axs = plt.subplots(int(np.ceil(len(list(y_hat.keys())) / 2)),2)
                    height = 3 * int(np.ceil(len(list(y_hat.keys())) / 2))
                    fig.set_figheight(height)
                    fig.set_figwidth(10)
                    if int(np.ceil(len(list(y_hat.keys())) / 2)) == 1 : 
                        for y_plot in range(0, len(list(y_hat.keys()))) :
                            pred = list(y_hat.keys())[y_plot]
                            matrix = np.concatenate([y_hat[pred][k][:,t] if (len(y_hat[pred][k][:,t].shape) > 1) else y_hat[pred][k][:,t].reshape(-1,1) for k in y_hat[pred].keys()], axis = 1 )
                            for idx in range(0, matrix.shape[0]) :
                                matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                            x = list(y_hat[pred].keys())
                            if plot_sample :
                                for idx in range(0, plot_sample) :
                                    axs[y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                            else :
                                for idx in range(0, matrix.shape[0]) :
                                    axs[y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                            avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                            axs[y_plot].plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                            axs[y_plot].set_title(pred)
                        plt.suptitle('ICE {}'.format(targets[t]))
                        plt.tight_layout()
                        if savedir :
                            i = 0
                            suffix = ''
                            savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            while os.path.exists(os.path.join(savedir, savename)) is True :
                                i = i + 1
                                suffix = i
                                savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            name = os.path.join(savedir,savename)
                            plt.savefig(name, bbox_inches = "tight")
                        if pdf :
                            pdf.savefig(bbox_inches = "tight")
                    else :
                        i = 0
                        for x_plot in range(0, int(np.ceil(len(list(y_hat.keys())) / 2))) :
                            for y_plot in range(0, 2) :
                                pred = list(y_hat.keys())[i]
                                matrix = np.concatenate([y_hat[pred][k][:,t] if (len(y_hat[pred][k][:,t].shape) > 1) else y_hat[pred][k][:,t].reshape(-1,1) for k in y_hat[pred].keys()], axis = 1 )
                                for idx in range(0, matrix.shape[0]) :
                                    matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                                x = list(y_hat[pred].keys())
                                if plot_sample :
                                    for idx in range(0, plot_sample) :
                                        axs[x_plot,y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                                else :
                                    for idx in range(0, matrix.shape[0]) :
                                        axs[x_plot,y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                                avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                                axs[x_plot,y_plot].plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                                axs[x_plot,y_plot].set_title(pred)
                                i = i + 1 
                                if i == len(list(y_hat.keys())) :
                                    break
                        plt.suptitle('ICE {}'.format(targets[t]))
                        plt.tight_layout()
                        if savedir :
                            i = 0
                            suffix = ''
                            savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            while os.path.exists(os.path.join(savedir, savename)) is True :
                                i = i + 1
                                suffix = i
                                savename = 'ICE_{}{}.png'.format(targets[t], suffix)
                            name = os.path.join(savedir,savename)
                            plt.savefig(name, bbox_inches = "tight")
                        if pdf :
                            pdf.savefig(bbox_inches = "tight")
    else :
        if plot_type == 'single' :
            for key in list(y_hat.keys()) :
                matrix = np.concatenate([y_hat[key][k] for k in y_hat[key].keys()], axis = 1 )
                for idx in range(0, matrix.shape[0]) :
                    matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                plt.figure(figsize = (16,9))
                x = list(y_hat[key].keys())
                if plot_sample :
                    for idx in range(0, plot_sample) :
                        plt.plot(x, matrix[idx, :], c = 'black', alpha = .1)
                else :
                    for idx in range(0, matrix.shape[0]) :
                        plt.plot(x, matrix[idx, :], c = 'black', alpha = .1)
                avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                plt.plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                plt.title('{}'.format(key))
                plt.tight_layout()
                if savedir :
                    savename = 'ICE_{}'.format(key)
                    name = os.path.join(savedir,savename)
                    plt.savefig(name, bbox_inches = "tight")
                if pdf :
                    pdf.savefig(bbox_inches = "tight")
        if plot_type == 'All' :
            fig, axs = plt.subplots(int(np.ceil(len(list(y_hat.keys())) / 2)),2)
            height = 3 * int(np.ceil(len(list(y_hat.keys())) / 2))
            fig.set_figheight(height)
            fig.set_figwidth(10)
            if int(np.ceil(len(list(y_hat.keys())) / 2)) == 1 : 
                for y_plot in range(0, len(list(y_hat.keys()))) :
                    pred = list(y_hat.keys())[y_plot]
                    matrix = np.concatenate([y_hat[pred][k] for k in y_hat[pred].keys()], axis = 1 )
                    for idx in range(0, matrix.shape[0]) :
                        matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                    x = list(y_hat[pred].keys())
                    if plot_sample :
                        for idx in range(0, plot_sample) :
                            axs[y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                    else :
                        for idx in range(0, matrix.shape[0]) :
                            axs[y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                    avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                    axs[y_plot].plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                    axs[y_plot].set_title(pred)
                plt.suptitle('ICE')
                plt.tight_layout()
                if savedir :
                    savename = 'ICE'
                    name = os.path.join(savedir,savename)
                    plt.savefig(name, bbox_inches = "tight")
                if pdf :
                    pdf.savefig(bbox_inches = "tight")
            else :
                i = 0
                for x_plot in range(0, int(np.ceil(len(list(y_hat.keys())) / 2))) :
                    for y_plot in range(0, 2) :
                        pred = list(y_hat.keys())[i]
                        matrix = np.concatenate([y_hat[pred][k] for k in y_hat[pred].keys()], axis = 1 )
                        for idx in range(0, matrix.shape[0]) :
                            matrix[idx,:] = matrix[idx, :] - matrix[idx, 0]
                        x = list(y_hat[pred].keys())
                        if plot_sample :
                            for idx in range(0, plot_sample) :
                                axs[x_plot,y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                        else :
                            for idx in range(0, matrix.shape[0]) :
                                axs[x_plot,y_plot].plot(x, matrix[idx, :], c = 'black', alpha = .1)
                        avg_val = [np.mean(matrix[:,col]) for col in range(0, matrix.shape[1]) ]
                        axs[x_plot,y_plot].plot(x, avg_val, c = '#eaf51b', linewidth = 4)
                        axs[x_plot,y_plot].set_title(pred)
                        i = i + 1 
                        if i == len(list(y_hat.keys())) :
                            break
                plt.suptitle('ICE')
                plt.tight_layout()
                if savedir :
                    savename = 'ICE'
                    name = os.path.join(savedir,savename)
                    plt.savefig(name, bbox_inches = "tight")
                if pdf :
                    pdf.savefig(bbox_inches = "tight")
    plt.close('all')
                        


def permutation_feature_importance(df, predict_fn, predict_fn_args = None, sample = None, predictors = None, prepare_df_fn = None, prepare_df_fn_args = None, compute_metric_fn = None,
                                   compute_metric_fn_args = None, figsize = None, plot = True, savedir = None, savename = None, pdf = None) :
    '''
    Permutation feature Importance Implementation

    - df : Dataframe di test
    - predict_fn : funzione di predict, esempio : model.predict
    - predict_fn_args : dizionario contenente argomenti extra della funzione di predict
    - predictors : lista features
    - prepare_df_fn : funzione da passare al df prima del predict (ad esempio associazione layer colonne)
    - prepare_df_fn_args : argomenti extra di prepare_df_fn
    - compute_metric_fn : funzione che prende in input il risultato del predict e da in output la metrica
    - compute_metric_fn_args : argomenti extra di compute_metric_fn
    - figsize = figsize del plot
    - savedir : cartella per salvare il plot
    - pdf : oggetto pdf per generare
    '''
    if prepare_df_fn :
        if prepare_df_fn_args :
            preprocessing = partial(prepare_df_fn, **prepare_df_fn_args)
        else : 
            preprocessing = partial(prepare_df_fn)
    if predict_fn_args :
        predict = partial(predict_fn, **predict_fn_args)
    else :
        predict = partial(predict_fn)
    if compute_metric_fn_args :
        compute_metric = partial(compute_metric_fn, **compute_metric_fn_args)
    else :
        compute_metric = partial(compute_metric_fn)
    if not predictors :
        predictors = d.columns
    if sample :
        if df.shape[0] > sample :
            d = df.sample(sample, random_state = 28)
        else :
            d = df.copy()
    else :
        d = df.copy()
    y_hat = {}  
    if prepare_df_fn :
        y_hat['str'] = compute_metric(predict(preprocessing(d)))
    else :
        y_hat['str'] = compute_metric(predict(d))
    for col in predictors :
        d_ = d.copy()
        np.random.shuffle(d_[col].values)
        if prepare_df_fn :
            y_hat[col] = compute_metric(predict(preprocessing(d_)))
        else :
            y_hat[col] = compute_metric(predict(d_))
    for k in list(y_hat.keys()) :
        if k != 'str' :
            y_hat[k] = np.abs(y_hat['str'] - y_hat[k])
    del y_hat['str']
    y_hat = {k : v for k,v in sorted(y_hat.items(), key=lambda item : item[1], reverse=False)}
    y = list(y_hat.keys())
    x = list(y_hat.values())
    if figsize : 
        plt.figure(figsize = figsize)
    else :
        h = int(np.ceil(len(predictors) * 0.5))
        plt.figure(figsize = (10, h))
    y_pos = np.arange(len(y))
    plt.barh(y_pos, x, align='center')
    plt.yticks(y_pos, y)  
    plt.xlabel('Importance')
    plt.title('Permutation Feature Importance')   
    plt.tight_layout()
    if savedir :
        if savename :
            name = os.path.join(savedir,savename)
            plt.savefig(name, bbox_inches = "tight")
        else : 
            print('please, specify a savename')
    if pdf :
        pdf.savefig()
    if plot == True :
        plt.show()
    plt.close('all')



def surrogate_model(X, Y, plot_y_distribution = False, figsize = None, plot = True, savedir = None, savename = None, pdf = None ) :
    '''
    Train a surrogate model on predictions using predictors as X.

    X = Dataframe with data and predictors
    Y = predictions
    plot_y_distribution = if True plot histogram of y
    figsize = size of resulting figure
    plot = if true show plot
    savedir = results directory
    savename = name of saved figure
    pdf = pdf object
    '''
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    r = results.rsquared
    pvalues = dict(results.pvalues)
    coeff = dict(results.params)
    for k in list(pvalues.keys()) :
        if pvalues[k] > 0.05 :
            del coeff[k]
    y_plot = list(coeff.keys())
    vals_neg = [v if v < 0 else 0 for v in coeff.values()]
    vals_pos = [v if v > 0 else 0 for v in coeff.values()] 
    xt_neg = np.linspace(max(vals_neg), min(vals_neg), 5)
    xt_pos = np.linspace(min(vals_pos), max(vals_pos), 5)
    xt_neg = [np.round(x,1) for x in xt_neg if x != 0]
    xt_pos = [np.round(x,1) for x in xt_pos]
    y_pos = np.arange(len(y_plot))
    if figsize :
        fig, axs = plt.subplots(1,2, sharey= True, figsize = figsize)
    else :    
        w = 0.25 * len(y_plot)
        fig, axs = plt.subplots(1,2, sharey= True, figsize = (8, w))
    axs[0].barh(y_pos, vals_neg, color = 'red')
    axs[1].barh(y_pos, vals_pos, color = 'blue')
    plt.subplots_adjust(wspace=0.0001, hspace=0.0001)
    axs[0].set_yticks(y_pos)
    axs[0].set_yticklabels(y_plot)
    axs[0].set_xticks(xt_neg)
    axs[1].set_xticks(xt_pos)
    axs[0].set_xticklabels(xt_neg)
    axs[1].set_xticklabels(xt_pos)
    plt.suptitle('Surrogate Model Coeff \nRsquared : {}\nJarque Bera on residuals pvalue : {}'.format(np.round(r,3), sms.jarque_bera(results.resid)[1]))
    if savedir :
        if savename :
            name = os.path.join(savedir,savename)
            plt.savefig(name, bbox_inches = "tight")
        else : 
            name = os.path.join(savedir,'surrogate_linear.png')
            plt.savefig(name, bbox_inches = "tight")
    if pdf :
        pdf.savefig()
    if plot == True :
        plt.show()
    if plot_y_distribution :
        plt.figure(figsize = (10,10))
        plt.hist(Y)
        plt.title('Y distribution')
    plt.close('all')


def plot_shapely(model, test_set, sample = None, show_plot=False, png_dir=None, title=None, figsize=(16, 9)):
    """Function that plots SHAP values for each feature.
    Arguments:
        model {tree-based model} -- input model. Non exhaustive list of accepted models: lightgbm, xgboost, catboost, scikit-learn RandomForest
        test_set {pandas DataFrame or numpy ndarray} -- dataset on which to evaluate features importances with shap algorithm
    Keyword Arguments:
        sample {int} -- sampled observation
        show_plot {bool} -- whether to show plot (default: {False})
        png_dir {str} -- path where static png file will be saved; if None, nothing will be saved (default: {None})
        title {str} -- title of the plot and part of filename (default: {None})
        figsize {tuple} -- figure size og the plot (default: {(16, 9)})
    """
    if sample :
        if test_set.shape[0] > sample :
            test_set = test_set.sample(sample, random_state = 28)
    log.info('Start evaluating shap values')
    shap_values = shap.TreeExplainer(model).shap_values(test_set)
    log.info('Finished evaluating shap values\n')
    fig = plt.figure(figsize=figsize, clear=True)
    if title is None:
        title = "Feature importances with SHAP values"
    shap.summary_plot(shap_values, test_set, show=show_plot, max_display=25, title=title, plot_size=None)
    if png_dir:
        file_path = os.path.join(png_dir, title + r'.png')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
    plt.close('all')

