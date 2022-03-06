# Credit_Risk_Analysis

## Overview and Objective

Over the last decade there has been a noticeable increase in personal lending. With such incredible growth, many lending firms and institutions are turning to machine learning techniques to continuously analyze large amounts of data and predict trends to optimize lending. Additionally, these firms are relying on these machine learning techniques to aid in predicting and identifying credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud. 

While credit risk can be categorize as belonging to class of supervised machine learning since the target variable (say bad loans, with high or low risk class) is known, it is an inherently unbalanced classification problem as good loans easily outnumber risky loans. 

The main goal of this project was to use Python to build and evaluate several machine learning models to predict credit risk for a given data set. Specifically, to employ imbalanced-learn and scikit-learn libraries to train and evaluate models with unbalanced classes using several resampling techniques, to determine which is better at predicting credit risk.

## Analysis

The following steps were used in the analysis.

1. The data was clean and prepared including converting the string variables into numerical values using the get_dummies() method.
2. The feature and target variables was created and checked.
3. The data was then separated into train and testing sets.

Since the dataset was unbalanced the following techniques were employed to create a more balance dataset. 

4. The data was oversampled using naive RandomOverSampler and SMOTE algorithms.
5. ClusterCentroids algorithms was used to undersample the data. 
6. A combinatorial approach of over- and undersampling using the SMOTEENN algorithm was also used. 
7. Additionally, two different ensemble classifiers that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, was used to predict credit risk.

The following steps(/methods) were employed to evaluate the performance of each algorithm listed above, and a comparison between models made where applicable. 

* The count of the target classes using Counter from the collections library was checked to ensure the dataset was balanced.
* The resampled data was used to train a logistic regression model.
* The balanced accuracy score from sklearn.metrics was calculated.
* The confusion matrix from sklearn.metrics was generated.
* A classication report using the imbalanced_classification_report from imbalanced-learn was generated. 


## Results

The metrics used to analyze the performance of the various models in this analysis is the balanced accuracy score from sklearn.metrics and the precision (pre) and recall (rec) scores from imbalanced-learn. Precision is the measure of how reliable a positive classification is. That is, it is intuitively the ability of the classifier not to label as positive a sample that is negative. Recall is the ability of the classifier to find all the positive samples. 

Figure 1 and 2 below shows the results of the oversampling methods, the naive random oversampling algorithm and the SMOTE algorithm, respectively. As can be observed from both figures the target variable (loan status) counter increased after fitting the model.

Figure 1. 

Figure 2. 

From Figure 1, Random Oversampling, the following observations and deductions can be made:

* With a moderate accuracy score of 0.645, the model is ok (at best) for predicting loan status (that is, high-risk or low-risk). 
* The precision for high risk performance is extremely low (0.01) indicating a large number of false positives, which indicates an unreliable positive classification. The recall is moderate (0.61) which indicates a low to moderate number of false negatives.
* In stark contrast, the precision for low risk is very good indicative of a low false positive rate. 

From Figure 2, SMOTE Oversampling, the model also had a moderate accuracy score of 0.643 indicating that the model is ok (at best) for predicting loan status. A comparison between the precision and recall metrics from Figures 1 and 2 (Random and SMOTE oversampling, respectively) shows the results are almost identical. Thus, the deductions discussed above for the precision and recall scores for Random sampling is also applicable for SMOTE.

Figure 3 below displays the results from the SMOTEENN algorithm. The model performance is almost identical to that of the Oversampling methods. The model also had a moderate accuracy score of 0.635 indicating that it is ok (at best) for predicting loan status. A comparison of Figures 1, 2 and 3 shows almost identical result for the accuracy scores, and the performance and recall scores thus the above mentioned deductions for random sampling is also applicable for the SMOTEENN algorithm.

Figure 3.


Figure 4 below displays the results from the Cluster Centroids undersampling algorithm. As can be observed from the counter the number of target variables decreased after fitting the model. The following deductions can be made:

Figure 4.

* This model is not good at classifying loan status because the model's accuracy, 0.516 is low.
* For high-risk class, the model has a moderate recall (0.63) but extremely low precision(0.01) indicating that most of its predicted labels are incorrect when compared to the training labels. 
* In contrast, for the low-risk class, the high precision (1.00) but low recall (0.43) results indicates the model returned very few results, but most of its predicted labels are correct when compared to the training labels. 

Figure 5 and 6 below display the results from the Balanced Random Forest Classifier (BRFC) and Easy Ensemble AdaBoost classifier (EEAC), respectively. Upon observation and comparison, the following deductions can be made:

Figure 5. 


Figure 6.



* Both ensemble learners are good at predicitng loan status with accuracy scores of 0.788 and 0.925 for the BRFC and EEAC, respectively. Of course, with an accuracy score of 0.925 the EEAC is better and it is very good.
* For both models, the precision is extremely low for high-risk class but very good for the low risk class, as observed in the other models above.
* The recall is very good for both classes (high-risk and low-risk) for the EEAC model which indicates a low number of false negatives.

## Summary

This project utilized several models to predict loan status for a dataset based on a set of feature variables. The key observations made are:

1. The two oversampling methods, random oversampling algorithm and SMOTE algorithm, and the SMOTEENN model performed approximately the same. There was no significant difference in the metrics (precision and recall) used to analyse the models. These models perfomed fairly good at predicting the loan status with accuracy scores of approximalty 0.64. However the average F1-score (subsequently discussed) of the Random oversampler (0.81) is the highest, therefore this model is the best of the three. 

2. The Cluster Centroid algorithm, that undersamples a given unbalanced dataset, had a low accuracy score and is therefore not good at classifying loan status. 

3. Both ensemble learners are good at predicitng loan status with accuracy scores of 0.788 and 0.925 for the BRFC and EEAC, respectively. Of course, with an accuracy score of 0.925 the EEAC is better.

It is difficult to compare two models with different precision and recall. Therefore, the F-score, which is the Harmonic Mean of precision and recall, is the metric that is often used to make different models comparable. Based on F1 scores, the best forming model is the Easy Ensemble Classifier with an average F1 score of 0.97. This model is therefore recommended for use to predict loan status.
