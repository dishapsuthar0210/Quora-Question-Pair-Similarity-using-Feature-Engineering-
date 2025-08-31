# Quora-Question-Pair-Similarity-using-Feature-Engineering-

Steps to generate the results.
Step 1. Clone the repo into local.
Step 2. Run the file "1_Quora_EDA_Basic_Feature_Extraction.ipynb"
Step 3. Run the file "2_Quora_Advanced_Feature_Enginnering_Visulization.ipynb"
Step 4. Run the file "3_Quora_Embedding_&_Feature_Engineering.ipynb"
Step 5. Run the "4_Quora_Model_Train_&_Results.ipnb"

##1_Quora_EDA_Basic_Feature_Extraction.ipynb : "Problem Statement & Basic Feature Extraction"
Work Flow starts from here.
1. Open the file in any python compatible environment 
2. Install necessary libraries and import
3. Load "Quora Question Pair" dataset which has 4Lakh+ such pairs using "train.csv"
4. After understanding the problem statement , run the file and you will be able to extract 11 basic features from given dataset.
5.This will generate extracted feature file , you need to save them and need to upload in next notebook.

##2_Quora_Advanced_Feature_Enginnering_Visulization : 
In this notebook, more 15 advanced NLP features will be extracted using "fuzzywuzzy".
-Also preprocessing of text will take place which will remove tags, punctuations.
-Performed stemming and removed stopwords in order to measure semantic similarity between two questions of any length.
-Extracted features will be analyzed and plotted their relation with each other to understand which features can be useful according to given problem statement.
-2D & 3D visualization is done using by t-SNE.

##3_Quora_Embedding_&_Feature_Engineering :
Converted into embedding and combined all useful features
-Converted each question into 384 size vector.
-11 basic features+15 advanced features+384 word embedding for question 1 + 384 word embedding for question 2
-Total  797 features extracted to train the model.
-All such features are saved in joblib files which will be created after running this notebook

##4_Quora_Model_Train_&_Results :
-Created database to store all extracted features.
-Preprocessing on that dataset.
-We have binary classification problem on large number of extracted features.
-Used log-loss as primary metric & confusion matrix as secondary metric.
-Trained Random model & got 0.88 log-loss.
-On Logistic Regression with hyperparameter tunning ,it  gave 0.4092 log loss on test data with best alpha 0.01. Train loss=0.4081
-On Linear SVM with hyperparameter tunning, it gave 0.4234 log loss on test data with best alpha 0.001. Train loss=0.4209
-On XGBoost, it gave 0.3445 log loss on test data.
-Plotted Confusion matrix, Precision Matrix & Recall Matrix after every model.
