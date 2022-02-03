# Credit Risk Analysis

## Overview of Analysis

The purpose of this project was to use supervised machine learning to analyze the risk of loans based on a variety of finanacial and personal information. The data for this analysis was in the LoanStats_2019Q.csv file. This data was analyzed using Pandas in Jupyter Notebook but specifially the scikit-learn and imbalanced-learn libraries to use supervised machine learning models. Various supervised machine learning models were tested to find the best fit for this data set.

### Supervised Machine Learning

In order to use machine learning models, the data has to be cleaned by removing any null values and convert data into the right data types and creating a key for the target data column ("loan_status") before converting to numerical values. Other columns were changed to numerical values since there can be no columns with string datatypes for machine learning models. The features for this model (X) included all of the columns but "loan_status" which was the target column (y). The data was split into training and testing groups. The model would be trained on the training data groups and then used to predict values on the testing groups. For any data set, there is a question of how to correctly sample from the data and there are multiple ways to sample from the data. The different methods can then be tested to see which method best predicts the data.

## Results

In order to see which sampling method created the best results, different methods were used, tested and the balanced_accuracy_score method to create a classification report to see which method performed the best.

1. Naive Random Oversampling

- The process of naive random oversampling is a method for an unbalanced data set where more samples from the smaller portion of the data are randomly added so that it is even with the larger portion of the data. The sampling method was used on the data and then used to run a logistics regression model. The accuracy of this sampling technique on the testing data, gave a score of a little over 65% (0.653397) meaning that this model could predict with that accuracy. The classification report showed that the model had a recall score of 67% (0.67). The classification report was generated and displayed in the image below.
![Naive Oversampling Classification Report](https://github.com/likenberry/Credit_Risk_Analysis/blob/main/Resources/Oversampled_Matrix.png)

2.SMOTE Oversampling

- The SMOTE (synthetic minority oversampling technique) oversampling method is another method used on unbalanced datasets. This method creates synthetic samples from the smaller portion of the data to create a more even dataset. The same process was used to run a logistic regression model with the SMOTE method and the accuracy and confusion matrix were generated. The accuracy of this method was only slightly less than the naive sampling method with a score of around 65% (0.651229) and the classification report showed that this method was better at predicting low risk applicants than high risk applicants. The recall score of 66% (0.66). The output of the confusion matrix is displayed below.
![SMOTE Oversampling Classification Report](https://github.com/likenberry/Credit_Risk_Analysis/blob/main/Resources/SMOTE_Matrix.png)

3.Undersampling

- Undersampling is another technique for dealing with an uneven data set but this time the larger portion of data is reduced to match the smaller portion of data. Another type of undersampling is cluster centroid undersampling which is similar to SMOTE where synthetic data points are created (centroids) representative of the entire cluster. The model was trained and fitted and the accuracy score, confusion matrix and classification report were created. The accuracy score was close to 53% (0.529302) and the classification report showed a recall score of 45% and the model was better at predicting the high risk applicants than the low risk applicants. The below image shows the output of the classification report.
![Undersampling Classification Report](https://github.com/likenberry/Credit_Risk_Analysis/blob/main/Resources/Undersampling_Matrix.png)

4.Combination (Over and Under) Sampling

- Another method tried on this dataset was a combination of over and undersampling, specifically SMOTEENN.  This process uses SMOTE sampling and ENN (Edited Nearest Neighbors) where the smaller portion of the dataset is sampled using SMOTE and the data is then cleaned and undersampled using ENN. The model was trained and fitted and the accuracy score, confusion matrix and classification report were generated. The accuracy score was almost 51% (0.516939) and the recall score from the classification report was 46% (0.46) but was better at predicting high risk applicants than loan risk. Below is the output of the confusion matrix.
![SMOTEENN Classification Report](https://github.com/likenberry/Credit_Risk_Analysis/blob/main/Resources/Combination_Matrix.png)

5.Balanced Random Forest Classifier

- Another way to create a better model is to use ensemble learning which combines multpile models to create an even stronger model. A type of ensemble model is a random forest model which uses a bunch of small tress or weak learners to build up to the best predictions. This model was created, trained and fitted and the accuracy score, confusion matrix and classification report was generated. The accuracy score was around 78% (0.6787124) but a recall score of 91% (0.91) and was better at predicting low risk than high risk. The below image shows the classification report output from the model
![Random Forest Classification Report](https://github.com/likenberry/Credit_Risk_Analysis/blob/main/Resources/Random_Forest_Matrix.png)

6.Easy Ensemble AdaBoost Classifier

- Another form of ensemble models is adaptive boosting (AdaBoost) where a model is run and the second model takes the errors of the first model to improve it until errors are minimized. The model was trained and tested and the accuracy and confusion matrix were generated. The accuracy score for this model was around 92% (0.925456) and the recall score from the confusion matrix was 94% (0.94). This model had the best accuracy of predicting both the high (0.91) and low (0.94) risk categories. The output image is below.
![AdaBoost Classification Report](https://github.com/likenberry/Credit_Risk_Analysis/blob/main/Resources/AdaBoost_Matrix.png)

## Summary

Overall, these models identified way more high risk loans than actually existed which is not a bad outcome since classifying more loans has high risk would protect the provider of the loan. However, it would make it harder for people who need loans to get loans if they are classified as high risk when they are not actually high risk. The various different methods of changing the model did change the accuracy however if one of these changes were actually to be implemented, the model with the best accuracy was the Easy Ensemble Classifier. This model had a score of 92% meaning that loans were correctly identified by their risk value 92% of the time. However, since these models were better at overidentifying high risk loans a lot of loans would be denied when it was not necessary. This is a good start to creating a supervised learning model to work with credit-risk data.

## Resources

- Data Source: LoanStats_2019Q1.csv
- Software: Visual Studio Code (Version 1.63.2), Jupyter Notebook, Pandas
