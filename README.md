[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com) ![](https://img.shields.io/badge/Software%20Mac->=10.14-pink)

# Deep Learning Text Classification 

## Classifing lines from the Game of Thrones into their correct seasons... can it be done? 

<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/stark_115482.png"/></div>

Convolutional neural networks (CNNs) working alongside word embeddings are like a match made in heaven for text classification. The word embeddings are able to hold semantic information about the word, while the CNN is able to model local structure about _how_ the word is used. Together, the two approaches are able to decipher quite a lot about the word in question and where it belongs in the classification problem... pretty big stuff! 

This assignment aimed to pull these two appraoches together and build a CNN with a dense word embedding layer, to classify lines from the popular Game of Thrones series. The model attempts to place the line into its correct season. To have a point of comparison, a logistic regression classifier is also built to see how this simpler classification approach will manage the complex dataset. 


## Table of Contents 

- [Assignment Description](#Assignment)
- [Scripts and Data](#Scripts)
- [Methods](#Methods)
- [Operating the Scripts](#Operating)
- [Discussion of results](#Discussion)

## Assignment Description

_This assignment is taken from “Assignment 4 –Deep Learning Text Classification” in the Language Analytics Summer 2021 course._

**Applying deep learning text classification to Games of Thrones episodes**

***Winter is... hopefully over.***

This assignment was interested in seeing how deep learning models like CNNs can be used for text classification purposes. It asked us to build models to classify a specific kind of cultural data - scripts from the TV series Game of Thrones.

Specifically, the assignment asked us to create two scripts to be run on the dataset:  
1.	A Logistic Regression classifier 
2.	A CNN classifier 


**Purpose of the assignment:** 

This assignment is designed to test that you have an understanding of:
1.	how to build CNN models for text classification;
2.	how to use pre-trained word embeddings for downstream tasks 
3.	how to work with real world, complex, cultural data 


**Personal focus of the assignment:** 
I took this assignment as an opportunity to dig into the theory behind deep learning CNN models and consider what all the different layers can do. To help my understanding, I played around with different layers, varying the size of the nodes and adding in drop out layers. For this reason, I have created a number of amendable parameters for the reader to experiment themselves with the data. It should however be noted that after many and deep attempts, I was never able to get the classifier above an accuracy of 40%, with most attempts ( such as the one documented through this article between 20 – 30% accuracy). 

## Scripts and Data 

**Scripts**

The assignment contains 2 scripts which can be found in the src folder: 

| Script | Description|
|--------|:-----------|
GoT_logistic_regression.py | Script containing the code to run the logistic regression classification  
GoT_CNN.py   | Script containing the code to run the convolutional neural network classification 

Each script contains detailed annotation of the steps used and so the reader is encouraged to look through it before running the script. 


**Data** 

The data used was a kaggle dataset of 23,911 lines taken across the 8 seasons of Game of Thrones. This can be downloaded from this [link](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons) or found in the data folder as Game_of_Thrones_Script.csv. 


## Methods 

### Logistic Regression 

To build the logistic regression classifier, I took inspiration from the documents we had covered in class. I first downloaded the Game of Thrones data and divided it into the text (named sentences) and the label (season). I then conducted a shuffled train-test split which would stratify the data equally, such that both the train and test set would have an equal representation of each season contained within (for example, if the training set was made up of 12% of season 8, then the test set would also be made up of 12% of season 8). For anyone who ended up watching the final season of Game of Thrones, we might agree that it would have been better for both our model and our minds if this season had simply never been released! The data then underwent vectorisation, which is a process of transforming the words into a numerical array. In this form, sentences are considered as vectors where each number is represented by a word in a vocabulary list. The training data in its numerical form was then passed into skLearn’s LogisticRegression function alongside the labels to train the classifier. The models performance was then tested on an unseen set of test data to see how well it would generalise to new sentences. The results of this model were visualised with a classification report and confusion matrix, which can be found in the output folder. 

### Convolutional Neural Network 

The convolutional neural network pre-processed the data in a similar way but added a number of steps to the pipeline before the data was fed into the model. As before, the data was divided into the text and labels variables and passed into the shuffled train_test split function, which divided the data into a distributed set of 80% train and 20% test data. The text was then vectorised into a numerical array and the labels were one-hot encoded to be compatible with our model using skLearn's LabelBinarizer function. Tensorflow's tokenizer with the number of words set to 5000 was then used to connect the text into sentences of numerical representation, within a defined volcabulary list. As the tokenized output has many differing lengths of the data, each sentences was padded with 0's to ensure that every sentence was the same length. This essentiall meant that every sentence (in its numerical representation) was now considered to be as long as the longest sentence in the dataset and 0's were used to represent words which were not there. An embedding matrix was then created with 50 dimensions, using GloVe's 50d set. The user is welcome to download and ammend this dimensional set to GloVe's 100 dimensional set using the parameter options. The kernel reularization value was set to kera's L2 with a value of 0.001 and then the model was ready to be built. This was organised with the following 6 layers: 

        1. The embedding layer
        2. The convulutional layer (128 nodes) 
        3. The GlobalMaxPool layer 
        4. The Dropout layer (0.01) 
        5. Dense layer of 32 neurons
        6. The prediction layer (of 8 nodes, 1 for each season) 

The model was compiled with the adam optimizer and metrics set to focus on accuracy, and trained on 15 epochs. The model was then evaluated using a classification report, confusion matrix, and accuracy plot. 

_As both scripts, particularly the latter CNN model, have a number of complex processing steps, it is highly recommended that the reader glances through the script before running it to see how the steps outline above connect together._


## Operating the Scripts 

***1. Clone the repository***

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository
git clone https://github.com/Orlz/Deep_Learning_Text_Classification.git

```

***2. Navigate into the repository***

After cloning, the user can navigate into the repository with the following code: 

```bash
$ cd Deep_Learning_Text_Classification

```


***3. Create the virtual environment***

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_virtual_environment.sh
```
And then activate the environment by typing:

```bash
$ source language_analytics04/bin/activate
```

***4. Run the logistic regression model***

### GoT_logistic_regression.py

Parameters = 2, required = 0 

```
Letter call  | Parameter      | Required? | Input Type     | Description                                         
-----------  | -------------  |--------   | -------------  |                                                     
`-t`         | `--test_size`  | No        | Float          | Number between 0 and 1 indicating the size of the test set                     
`-o`         | `--output_path'| No        | String         | Path to where the output should be stored    
```

The script can be run from the terminal with the following code: 


```bash
$ python3 src/LR_Text_Classification.py
```

***4. Run the Convolutional Neural Network model***

### GoT_CNN.py

Parameters = 4, required = 0

```
Letter call  | Parameter         | Required? | Input Type     | Description                                         
-----------  | ----------------  |--------   | -------------  |                                                     
`-g`         | `--glove`         | No        | String         | Path to the GloVe text file                   
`-o`         | `--output_path'   | No        | String         | Path to the output directory 
`-t`         | `--test_size`     | No        | Float          | Number between 0 and 1 indicating the size of the test set           
`-e`         | `--embedding_dim` | No        | Integer        | Number specifying the number of embedding dimensions to use   
```

The script can be run from the terminal with the following code: 


```bash
$ python3 src/GoT_CNN.py
```

## Discussion of Results 

The most important considerations when evaluating classification models is the F1 score. This score gives a more roundied insight into both precision and recall by being calculated as a harmonic mean of the two. The models history plot is also useful to consider as it can give an indication of how the model is performing over the number of runs and whether it is over or under fitting. Let’s take a look at each model in turn: 

**Logistic Regression**

The logistic regression model was able to perform with an F1 score of 0.27, which was around 14% above chance level (12.5%). This is a fair performance, considering the data has not been heavily pre-processed before being fed into the model, however it is not a result which we can get particularly excited about. The model was able to perform best on season 7 and not well at all on season 8… could this be a suggestion that much of our frustration from Season 8 was that it simply wasn’t able to differentiate itself enough? The predictions for each season are displayed below: 

 <img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/LR_Metrics.png" alt="alt text" width="500" height="400"> 
 
 Future attempts at the model are likely to benefit from splitting the data into chunks or classifying on a different label, such as using the character instead of the season.
 
 **Convolutional Neural Network**
 
Our CNN model was able to perform with an F1 score of 0.24, which is surprisingly worse than our logistic regression. Usually, deep neural networks are able to classify with much higher accuracies due to the addition of word embeddings and the colvoluted layers. Indeed, it is much more common for us to see accuracy results of at least above 50 with CNNs. What this low accuracy shows is that our model seems to be unable to learn well from the data it is being fed. The model accuracy plot can help us to get better insights into this: 

 <img src="https://github.com/Orlz/Deep_Learning_Text_Classification/blob/main/Output/CNN_modelhistory.png" alt="alt text" width="500" height="400"> 

Here we see the problem, the green validation accuracy is a flat line along the bottom, meaning the model is not being able to take it's learning from the training set and apply it into the test set. This flat line suggests that some of the accuracy above chance could be simply luck. Our training loss curve (blue) does descend, giving the impression that it's learning (albeit slowly), however the diverging validation loss curve (red) which continues to rise rather than fall is a strong indication that the model is overfitting this training data. All in all the results seem, as one might say, "nothing to write home about" ;) 

We can conclude from these results that classifying lines from Game of Thrones into their seasons is a particularly difficult task to achieve. This might be good news for the director's, meaning they have a series which flows seemlessly through each season! However, there are considerations which could help. A starting point would be to pre-process the data in a way that it would be more intepretable to the model. This could include chunking the data into sentences of 10, to give the model a bit more context to work with. The data could also be collected into chunks of according to the character who is speaking, going with the assumption that it is the characters which develop and change through the seasons rather than the whole show itself. Another way to get more impressive results would be to change the classification labels, from seasons to perhaps main characters instead. This would mean we'd be looking at whether the model would be able to predict the charaacter speaking across the whole series rather than season. There is one big flaw with this latter approach which we mustn't forget before diving into it... with the rate of murder through Game of Thrones, we'd be struggling to find characters who were alive long enough to be classified!! 


I hope you've enjoyed the explorations through deep learning classification 😊

___Teaching credit___ 

Many thanks to Ross Deans Kristiansen-McLachlen for providing an interesting and supportive venture into the world of Language Analytics! 

<div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>









<a href="https://iconscout.com/icons/house" target="_blank">Stark wolf Icon</a> by <a href="https://iconscout.com/contributors/roundicons-com">Roundicons .com</a> on <a href="https://iconscout.com">Iconscout</a>
