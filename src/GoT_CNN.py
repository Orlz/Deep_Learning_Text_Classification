#!/usr/bin/env python

"""
==================================================================
Assignment 6: Text classification using Deep Learning (CNN models)  
==================================================================

This script builds a deep learning text classification model by combining word embeddings with a convolutional neural network. 
The model is used to classify lines from the popular HBO series "Game of Thrones" into the season which the line belongs to. It
uses a kaggle dataset of 23,911 lines taken across the 8 seasons of Game of Thrones. This dataset can be found on kaggle at the
following address: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons.

Combining word embeddings and CNN approaches should enable us to capture local structure in the data which can be generalized
across the whole dataset. That is the hope in any case! 

This script takes the following 12 steps: 
    1.   Load and preprocess the data 
    2.   Create the training and validation set 
    3.   Vectorize the data using CountVectorizer()
    4.   Binarize the labels 
    5.   Word Embeddings  
    6.   Pad the sentences 
    7:   Define an embedding layer size and create an embedding matrix
    8.   Set the kernal reularization value
    9.   Build the model 
    10.  Compile the model 
    11.  Train and evaluate the model
    12.  Create and save an accuracy plot
    13.  Predictions and classification 
    
    
The script has the following argparse parameters: 
    -o  --output_path     String variable defining where the output directory should be (default: ../Output/)
    -g  --glove           String variable defining where the GloVe word embedding file is saved (default: data/glove.6B.50d.txt)
    -e  --embedding_dim   Integer number specifying the number of embedding dimensions to use (default: 50)
    -t  --test_size       Float number indicating the size of the test set (default: 0.2)
    

Usage:
  $ python3 Deep_Learning_Text_Classification.py
    
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
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

# Import tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, Flatten, GlobalMaxPool1D, Conv1D, Dropout)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# Import matplotlib for plots 
import matplotlib.pyplot as plt

"""
===============================
----- Argparse Parameters -----
===============================
"""

# Initialize ArgumentParser class
ap = argparse.ArgumentParser()

# Argument 1: Output directory 
ap.add_argument("-o", "--output_path", 
                type = str,
                required = False,
                help = "[INFO] Path to the output directory",
                default = "Output/")
    
# Argument 2: Glove file path 
ap.add_argument("-g", "--glove", 
                type = str,
                required = False,
                help = "[INFO] Path to the GloVe text file (default GloVe with 50 dimensions).",
                default = "data/glove.6B.50d.txt")

# Argument 3: Determining number of embedding dimensions  
ap.add_argument("-e", "--embedding_dim", 
                type = int,
                required = False,
                help = "[INFO] Integer number specifying the number of embedding dimensions to use (default: 50) ",
                default = 50)

# Argument 4: test size  
ap.add_argument("-t", "--test_size",
                type = float,
                required = False,
                help = "[INFO] Float number indicating the size of the test set (default: 0.2).",
                default = 0.2)

# Parse arguments
args = vars(ap.parse_args()) 


"""
=========================
----- Main Function -----
=========================
"""

def main():
    """
    ========================
    Functions through script
    ========================
    """
    
    def plot_history(H, epochs, output_path):
        
        """
        Ultility function for plotting the model history using matplotlib. 
            H:        model history
            epochs:   number of epochs for which the model was trained    
        """
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "CNN_modelhistory.png"))
    
    
    def create_embedding_matrix(filepath, word_index, embedding_dim):
        """ 
        A helper function to read in saved GloVe embeddings and create an embedding matrix
        (Many thanks to Ross for his creation) 
    
        filepath: path to GloVe embedding
        word_index: indices from keras Tokenizer
        embedding_dim: dimensions of keras embedding layer
        """
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        with open(filepath) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]

        return embedding_matrix

    """
    ===================
    Argparse Parameters 
    ===================
    """
    output_path = args["output_path"]
    glove = args["glove"]
    test_size = args["test_size"]
    
    #make output directory if one does not already exist
    if not os.path.exists(output_path):   
        os.mkdir(output_path)
    
    print("\nWe're all set - let's start some deep learning text classification") 

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
    print("\nThe data's loaded, now we'll split it up and do some moulding to turn it into a numerical array") 
    
    # We create a test set with the size defined from the command line
     
    X_train, X_test, y_train, y_test = train_test_split(sentences,                #The text we want to classify with 
                                                    season,                       #The season label we're using 
                                                    test_size=test_size,          #The test size defined in command line
                                                    random_state=42,              #The random state makes the sets reproducible
                                                    shuffle = True,               #Shuffle the data from the order it's fed in
                                                    stratify = season)            #Ensure even splits of data across seasons
    
    #save the label names for later 
    labelnames = sorted(set(y_train))
    
    
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
    
    # Finally, we create a list of the feature names. 
    feature_names = vectorizer.get_feature_names()
    
    """
    ===========================
    Step 4: Binarize the labels
    ===========================
    """
    
    # Binarize labels (i.e. the season labels) 
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    
    """
    ================================================
    Step 5: Tokenize the text into a numerical array
    ================================================
    """
    """
    Tokenizing is the process of coverting text into numbers - Tensorflow keras has a function to do this for us.
    What we do is essentially take the words and index them with a number from a defined volcabluary list. 
    This means that a sentence such as "Let's kill him" would now be represented by something like [278, 640, 42]
    """
        
    # Initialize a tokenizer with 5000 words (gets the full sentences)
    # Fit this to the x_train data 
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    # Then make the tokens into a sequence 
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)
    
    #Correct any possible index issues by adding 1 to the length of the tokenizer 
    vocab_size = len(tokenizer.word_index) + 1  # This helps counter the 0 index in python 

    # Let the user know your text has been tokenized 
    print (f"Your data has been tokenized!") 
    print (f"Your trained data looks like: {X_train[2]} ") 
    print (f"Your tokenized data now looks like: {X_train_toks[2]} ") 
    
    """
    ==========================
    Step 6: Pad the sentences 
    ==========================
    """
    """
    The tokenized output has many differing lengths of the data (as sentence length varies considerably).
    To overcome this, we can pad the sentences to make them of equal length. We do this by standardizing the length 
    of each sentence to a max_length and adding 0s to fill where characters are missing
        
    We'll find the maximum sentence length in both the training and test sets and match them accordingly
    """
    
    # max length for sentences in both training and test sets 
    train_len = len(max(X_train_toks, key=len))
    test_len = len(max(X_test_toks, key=len))
    maxlen = max(train_len, test_len)         

    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                            padding='post', # we're adding the sequence post instead of pre
                            maxlen=maxlen)
    
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                           padding='post', 
                           maxlen=maxlen)
    
    
    print("Padding complete. We're about to embed some layers") 
    
    
    """
    =====================================================================
    Step 7: Define an embedding layer size and create an embedding matrix 
    =====================================================================
    """
    """
    Word embeddings allow us to create dense numerical representations of our volcabulary which can hold semantic information.
    They work with the assumption that words which occur together frequently have similar meanings
        
    Here we use a pre-trained static word embedding model known as GloVe to map our volcabulary onto. 
    GloVe comes with differing ranges which the user is welcome to test on the model. The default has been set to glove50

    """
    
    # Embedding layer size 
    embedding_dim = args["embedding_dim"]
    
    #Embedding matrix
    embedding_matrix = create_embedding_matrix(glove,
                                               tokenizer.word_index, 
                                               embedding_dim)
    
    """
    ==========================================
    Step 8: Set the kernel reularization value   
    ==========================================
    """
    
    #We're using keras' L2 
    l2 = L2(0.001)
        
    """
    =======================
    Step 9: Build the model   
    =======================
    """
    """
    This begins with setting a base model
    """
    print("\nWe'll all set, not to build the model") 
    
    #Ensure old session is cleared 
    K.clear_session()
    
    # Create a sequential model 
    model = Sequential()
    
    
    """
    Next, the various layers are added to the model in turn.  
        1. The embedding layer
        2. The convulutional layer (128 nodes) 
        3. The GlobalMaxPool layer 
        4. The Dropout layer (0.01) 
        5. Dense layer of 32 neurons
        6. The prediction layer (of 8 nodes, 1 for each season) 
            
    """
    # 1. Embedding layer (using the embedding_matrix) 
    model.add(Embedding(vocab_size,                  # Vocab size from Tokenizer() (5000) 
                        embedding_dim,               # Embedding input layer size (50) 
                        weights=[embedding_matrix],  # Pretrained embeddings
                        input_length=maxlen,         # Maximum length of the padded sentences (100) 
                        trainable=True))             # The embeddings are trainable
    
    #2. Convulutional layer (using "relu" with a kernal regularizer) 
    model.add(Conv1D(128, 5, 
                    activation='relu',
                    kernel_regularizer=l2)) 
    
    #3. Global MaxPool layer (this is used instead of the flatten layer as it can reduce noise and increase signals within data)
    model.add(GlobalMaxPool1D())
    
    #4. Dropout layer (this forces the model to generalise by randomly dropping some of the information being passed forward) 
    model.add(Dropout(0.01))
    
    #5. Dense layer of 32 neurons (activation = relu) 
    model.add(Dense(32, activation='relu', kernel_regularizer=l2))
    
    #6. Prediction layer 
    model.add(Dense(8, activation='softmax'))
    
    """
    ==========================
    Step 10: Compile the model   
    ==========================
    """
    """
    We're using a categorical loss function because we have 8 categories we are trying to classify between. 
    Optimization algorithm = adam 
    """
    # Compile model 
    model.compile(loss='categorical_crossentropy', 
                  optimizer="adam",
                  metrics=['accuracy'])
    
    # ... and get the summary 
    summary = model.summary()
    
    #print this to the terminal 
    print(summary) 
    
    
    """
    =====================================
    Step 11: Train and evaluate the model    
    =====================================
    """
    """
    Now that the model is built, we want to run (train) it and evaluate how it's performing. We do this in 2 steps: 
        1. Fit the model 
        2. Evaluate the performance 
    """
    
    print("\nNow we're ready to train the model...") 
    
    #Fit the model (we'll run it through 10 epochs) 
    history = model.fit(X_train_pad, y_train,
                    epochs=15,
                    verbose=False,
                    validation_data=(X_test_pad, y_test),
                    batch_size=10)
    
    #Evaluate the model and print results to the terminal
    #Training data 
    train_loss, train_accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    
    #Test data
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(test_accuracy))
    
    """
    =========================================
    Step 12: Create and save an accuracy plot     
    =========================================
    """
    print("\nTraining complete, not to create some accuracy plots") 
    """
    We can visualise the performance of the model using an  accuracy plot
    """
    epochs = 15
    
    #Create plot in a figure called accuracy_plot - this will save to the output folder
    accuracy_plot = plot_history(history, epochs, output_path)

    
    """
    =========================================
    Step 13: Predictions and classification      
    =========================================
    """
    """
    Finally, we want to test how good our model is with predictions. 
    We'll create predictions and use them to build a classification report. 
    """
        
    # Get predictions:
    predictions = model.predict(X_test_pad)
        
    #Save the classification report as a text file
    report = classification_report(y_test.argmax(axis = 1),                                           
                                   predictions.argmax(axis = 1),
                                   target_names = labelnames, 
                                   zero_division = 0,
                                   output_dict = True)
    
    report_path = os.path.join(output_path, "CNN_Metrics.txt")
    with open(report_path, "w") as output_file:
        output_file.write(f"Classification Metrics from Convolutional Neural Network:\n{report}")
        
        
    #Create a confusion matrix from the predictions
    clf.plot_cm(y_test.argmax(axis = 1),
                predictions.argmax(axis = 1), 
                normalized=True)
    
    #Save the confusion matrix 
    matrix_path = os.path.join(output_path, "CNN_ConfusionMatrix.png")
    plt.savefig(matrix_path) 

    print("That's you all complete - your results are waiting for you in the Output folder! :)") 
    

if __name__=="__main__":
    #execute main function
    main()