#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:30:18 2018

"""

from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd 
import sys
from genetic_selection import GeneticSelectionCV


def main():
    myTrain = pd.read_csv('/Users/rajathanda/Downloads/Principal_Project/Principal_Data/Pr_train_20.csv')
    #myTest = pd.read_csv('/home/ajob2/Downloads/Pr_test_20.csv')
    #myTrain['Profit_loss'] = 1
   # myTest['Profit_loss'] = 1
    #myTrain.loc[myTrain['Excess'] <= 0,'Profit_loss'] = 0
   # myTest.loc[myTest['Excess'] <= 0,'Profit_loss'] = 0
    #myTrain.drop(['Excess','IDENTIFIER','PERIOD (FORMATTED)','FUTURE 24 WEEK RETURNS'], axis=1,inplace =True)
   # myTest.drop(['Excess','IDENTIFIER','PERIOD (FORMATTED)','FUTURE 24 WEEK RETURNS'], axis=1,inplace =True)
    cols = [col for col in myTrain.columns if col not in ['Profit_loss']]
    #cols_2= [col_2 for col_2 in myTest.columns if col_2 not in ['Profit_loss']]
    X_train = myTrain[cols]
    y_train = myTrain['Profit_loss']
    #X_test =  myTest[cols_2]
    #y_test =  myTest['Profit_loss']

    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(X_train), 20))

    X = np.hstack((X_train, E))
    y = y_train
    print("Hello")
    estimator = linear_model.LogisticRegression()

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy", # use f1 if supported
                                  n_population=100,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=100,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(selector.support_)

    with open("/Users/rajathanda/Downloads/Principal_Project/Principal_DataGenetic_Output",'w') as f:
    	f.writelines("----------Output----------------")
    	f.writelines(selector.support_)
    	f.writelines("-------------END----------------")

if __name__ == "__main__":
    main()
    
    
    
