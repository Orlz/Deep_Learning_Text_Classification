#!/usr/bin/env python

"""
===========================================================================
Assignment 6: Text classification using Deep Learning (Logistic Regression) 
===========================================================================

This script uses Logistic Regression to classify lines from the popular HBO series "Game of Thrones" into the season which the line belongs to. It uses a kaggle dataset of 23,911 lines taken across the 8 seasons of Game of Thrones. This dataset can be found on kaggle at the following address: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons

The script takes the following steps: 
1. Load and preprocess the data 
2. Create the training and validation set 
3. Vectorize the data using CountVectorizer() 
4. Build the Logistic Regression classifier  
5. Evaluate the performance
6. Create a confusion matrix 
7: Perform cross validation and visualise this using learning curves and models of scalibility and performance.


Optional parameters: 
    -t  --test_size     Float number indicating the size of the test set (default: 0.2)
    -o  --output_path   String variable defining where the output directory should be (default: ../Output/)
    
    
The script can be run from the command line by navigating to the correct directory and environment, then typing: 
    $ python3 LR_Text_Classification.py 
    
""" 

"""
=================================
----- Import Depenendencies -----
=================================
"""

# operating system tools
import os
import sys
import argparse
sys.path.append(os.path.join(".."))


# pandas, numpy, and gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import classification utilities and functions
import DL_Text_Classification.utils.classifier_utils as clf

# Import the machine learning tools from sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# Import matplotlib for plots 
import matplotlib.pyplot as plt


"""
===============================
-----  Argparse Commands  -----
===============================
"""

# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: test size  
ap.add_argument("-t", "--test_size",
                type = float,
                required = False,
                help = "[INFO] Float number indicating the size of the test set (default: 0.2).",
                default = 0.2)

# Argument 1: output directory  
ap.add_argument("-o", "--output_path",
                type = str,
                required = False,
                help = "[INFO] String variable defining where the output directory should be (default: ../Output/).",
                default = "Output/")


# Parse arguments
args = vars(ap.parse_args()) 


"""
=========================
----- Main Function -----
=========================
"""

def main():
    
    """
    ----------------------------
    Link the argparse parameters
    ----------------------------
    """
    test_size = args["test_size"]
    output_path = args["output_path"]
    
    # If the output directory doesn't exist yet, create one
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    """
    =====================================
    Step 1: Load and preprocess the data 
    =====================================
    """
    
    # Create a pandas dataframe with the Game of Thrones data 
    filepath = "data/Game_of_Thrones_Script.csv"
    df = pd.read_csv(filepath)
    
    # Reduce the dataframe to include only the necessary columns ('Season' and 'Sentence') 
    df = df[["Season", "Sentence"]]
    
    #Get the values from each cell of this dataframe into two lists
    season = df['Season'].values
    sentences = df['Sentence'].values
    
    
    """
    ==============================================
    Step 2: Create the training and validation set 
    ==============================================
    """
    # We create a test set with the size defined from the command line
    # The random state of 42 makes the train and test sets reproducible ('hitchhikers guide to the galaxy' ref ;)
    # Shuffle and stratify ensure the data is not fed into the model in the order it appears in the csv 
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                    season, 
                                                    test_size=test_size, 
                                                    random_state=42,
                                                    shuffle = True,
                                                    stratify = season)
    
    
    """
    ==================================================
    Step 3: Vectorize the data using CountVectorizer() 
    ==================================================
    """
    # Sklearn's count vectorizer creates a numerical vector from the input words
    vectorizer = CountVectorizer()   
        
    # We apply this vectorizer to the trainng data to make this into a numerical vector representation 
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then we do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    
    # Finally, we save the feature names from the vector
    feature_names = vectorizer.get_feature_names()
    
    
    """
    ==================================================
    Step 4: Build the Logistic Regression classifier  
    ==================================================
    """
    
    # Build the classifier 
    classifier = LogisticRegression(random_state=42, max_iter=100)
    
    #Train the classifier 
    classifier.fit(X_train_feats, y_train)  
    
    # Create predictions 
    y_pred = classifier.predict(X_test_feats)
    
    
    """
    ================================
    Step 5: Evaluate the performance   
    ================================
    """
    #Create a classification report and print it to the terminal
    report = metrics.classification_report(y_test, y_pred)
    print(report)
    
    #Save the ourput as a txt file 
    report_path = os.path.join(output_path, "LogisticRegression_Metrics.txt")
    with open(report_path, "w") as output_file:
        output_file.write(f"Classification Metrics from Logistic Regression:\n{report}")
    
    """
    =================================
    Step 6: Create a confusion matrix    
    =================================
    """
    #Confusion matrix 
    clf.plot_cm(y_test, y_pred, normalized=True)
    
    #Save the confusion matrix 
    matrix_path = os.path.join(output_path, "LogisticRegression_ConfusionMatrix.png")
    plt.savefig(matrix_path)
    
    
    """
    ================================
    Step 7: Perform cross validation 
    ================================
    """
    
    # Vectorize full dataset
    X_vect = vectorizer.fit_transform(sentences)

    # initialise cross-validation method
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # run on data
    model = LogisticRegression(random_state=42)
    accuracy_plot = clf.plot_learning_curve(model, title, X_vect, season, cv=cv, n_jobs=4)
    
    #save the plot 
    accuracy_plot_path = os.path.join(output_path, "LogisticRegression_accuracy.png")
    plt.savefig(accuracy_plot_path)
    


if __name__=="__main__":
    #execute main function
    main()
