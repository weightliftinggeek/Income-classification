# Problem Description

The dataset (income.csv) is collected and provided by the US Census database.

Develop 2 end-to-end classification models: Logistic Regression and SVM. Then, apply the K-Means clustering method to further understand the dataset.

# Task 1: Load the dataset, do basic data pre-processing, and split the dataset.
The pre-processed data has 10 columns and 26215 rows. The first column, income, is the target column with Boolean value of 0 and 1. There are 2 numeric variables which are age and hours-per-week. The rest are all categorical variables. Variable ‘martial-status’ and ‘hours-per-week’ were renamed to ‘marital_status’ and ‘ hours_per_week’ because the hyphen in variable name can cause code errors later on.  

There are 1396 missing values in workclass and 1401 missing values in occupation. This is proportionally small compared to the 26215 rows in total so we can drop these rows and still make sure there’s enough data in the dataset. Furthermore, 3277 duplicated rows (proportionally small compared to the total dataset) were also removed from the dataset. This ensures that the dataset is unique. After dropping duplicated rows and rows with missing values, there are 21537 rows left for further analysis. 21537 rows are considered plentiful for analysis and machine learning.  

Some variables originally are not in the right types, for example, ‘income’ was originally type integer while it’s supposed to be categorical. Datatype converting step was performed so that only ‘age’ and ‘hours_per_week’ are type integer.  

The descriptive statistic for ‘age’ variable is that age ranges from 17 to 90 with average age being 39, median 38. ‘hours_per_week’ ranges from 1 to 99 with average working hours being 41 and median working hours being 40.  
Correlation charts show that there is no clear correlation between ‘age’ and ‘hours_per_week’.  
Out of 7 categorical variables, only ‘education’ is ordinal variables where strings were converted to numbers in order between 1 and 16. This was done because there is hierarchy in the highest education achieved by a person.  
The rest of the nominal categorical variables were transformed to dummy variables where each value of a category becomes a variable and a boolean value of 0 or 1 assigned to the variable. This was performed to form the two-dimensional binary matrix that helps with machine learning’s performance (geeksforgeeks, 2020).  

Since sex is binary variable, we can use 0 to represent Male and 1 to represent Female.  
The final transformed data has 41 columns and 21537 rows.  
The first column, income, is defined as a target variable and the rest were input variables.  
Finally, the dataset is split so that 10% is used for testing and 90% used for training. The numeric variables are normalised so that features are of similar scale which reduces bias in the result.  
However, it is arguable if normalisation makes much difference in this dataset since ‘age’ and ‘hours-per-week’ are of similar scale already.  

# Task 2: Train and evaluate the 2 classification models on the training set with the cross-validation method, optimize the models and evaluate models on the test set.

Logistic regression and SVM are both used for classification tasks.  
Logistic regression uses the probability from the sigmoid function to classify the dataset. Value between 0 and 1 is assigned to any input value of this function. This probability value helps determining whether an input belong or not belong to a certain class. The threshold for the probability (decision boundary) is decided by the algorithm to determine which class a data point belongs to. For example, the data point will belong to class A if the probability is more than 50%, otherwise it will be classified as class B. The Cross-Entrophy function is used as a cost function to find the absolute minimum, hence define a model with minimum error (Aysh Pant, 2019).  

![image](https://github.com/user-attachments/assets/d68119bd-de19-41fb-b6da-6bb60e0292a9)

Support Vector Machines classify data points by a separating hyperplane. In other words, the algorithm is to make a line to best discriminate clusters of data for the classification task. The aim of SVM is to maximise the margin (the smallest distance between the line and the closet data point) using the hinge loss cost function (Rohith Gandhi, 2018).  
![image](https://github.com/user-attachments/assets/075c1ba5-33f0-4f0f-a32c-327dc7ac4d4c)
![image](https://github.com/user-attachments/assets/ceda1aed-9fb6-4aec-8360-938dbed52fa8)  
The testing accuracy of the model (default settings) based on the trained model using the entire training dataset (90% of the original dataset) is shown below:  
![image](https://github.com/user-attachments/assets/0e92eb52-b3be-4911-a18b-3c62362b961e)
Here SVM with 78.8% accuracy performs slightly better on the testing dataset compared to logistic regression 79.2% accuracy.    
After defining a 10 fold cross validation, the average accuracy of each trained model based on the training dataset is shown below:  
![image](https://github.com/user-attachments/assets/c3ae6e60-d334-45ac-a7da-ed8ddf4e6d4f)  
Here, logistic regression (lr) performs better with 80.8% accuracy on the training dataset as supposed to 79.9% accuracy by SVM. Also, the average accuracy of the cross validation on the training dataset is slightly higher than the testing accuracy for both models, which is expected.  

The three parameters selected when optimising the logistic regression model are: penalty, C and Solver (Scikit learn, logistic regression). For ‘penalty’, there is a choice between L1 and L2 regularisation. L2 uses “square magnitude” for its error term, whereas L1 uses “absolute value of magnitude”. L1 is favored for dataset with many variables because it filters out less important features (Anuja Nagpal, 2017).  

The options selected for ‘C’ parameter are 0.01, 0.001 and 0.1 According to Scikit learn, the lower the value, the higher level of regularisation. The default setting is 1. Initially, values larger than 1 was tested but there were convergence errors and long time for the code to complete running. It was assumed that with value larger than 1, the level of regularisation was too low and overfitting issue presented, which led to the convergence error. The error disappeared with higher regularisation.  

The options selected for ‘solver’ parameter (algorithm used to optimise the model) are ‘saga’ and ‘liblinear’. According to Scikit learn, ‘saga’ is better and faster for larger dataset. In this case, the Figure 3 SVM, margin - retrieved from Rohith Gandhi, 2018 optimisation has selected ‘saga’ over ‘liblinear’ for the dataset set that has 21537 rows and 41 variables. The algorithm has also decided that c = 0.1, penalty = l1 are best options for this model.  

The four parameters selected when optimising the SVM model are: kernel, C, degree and gamma (Scikit learn, SVM). ‘kernel’ is used to specify the kernel type in the algorithm. This is used to transform low dimensional input to higher dimensional space which helps the algorithm in classification (Velocity, 2020). ‘C’ is the level of regularisation which is similar to logistic regression but l2 only applied (Scikit learn, SVM), a choice of 1 and 10 is presented. According to Velocity 2020, parameter ‘gamma’ determines to what extent nearby and far away points have on the decision boundary (Velocity, 2020), a choice of ‘auto’ and ‘scale’ is presented. ‘degree’ represents the degree of polynomial kernel function (Scikit learn, SVM), here there is a choice of 3 or 8. The algorithm selected C = 10, degree = 3, gamma = scale and kernel = ‘poly’ as best options for this model.  

It is worth noticing that the gradient descent for logistic regression has linear convergence whereas SVM has sub-linear convergence. This may explain why it took a lot longer for the code to complete running when fine tunning the SVM model.  
The accuracy based on the training dataset for each model before and after fine tuning are  
![image](https://github.com/user-attachments/assets/c4d3104a-c867-41d6-a07d-20265655d705)  
In this case of logistic regression, optimisation hasn’t improved the accuracy on the training dataset.   
![image](https://github.com/user-attachments/assets/061ba69d-c89b-4134-9dd3-47c1d8826a37)  
In this case of SVM, optimisation has improved the accuracy on the training dataset by 0.5%.  

The accuracy based on the testing dataset for each model after fine tuning are  
![image](https://github.com/user-attachments/assets/77fee385-f0f1-45fb-8040-655be985b98c)  

Despite the fine-tuned lr model performing slightly better than the SVM fine-tuned model on the train set (80.7% vs 80.6%), the SVM fine-tuned model performed 0.5% better than the fine-tuned lr model on the test set (78.9% vs 79.4%). These results are slightly higher than the testing accuracy prior to fine tuning the model, 78.8% for lr and 79.2% for SVM. In brief, combing model fine tuning and cross validation provide better accuracy on the dataset. There is not much different in performance between lr and SVM on this dataset.  

The summarised table of all results are below:  
![image](https://github.com/user-attachments/assets/052f77dc-09cd-4c4c-bdaa-851a1e307938)  

# Task 3: Apply K Means clustering on the normalised training input X, and understand the grouping of training data by investigating the prototype.

K mean clustering is an unsupervised machine learning technique for classifying data, preferably used when labels are not known. Like logistic regression and SVM, K mean clustering aims to group similar data points together. After specifying the number of clusters/centroids that the data should be classified/belong to, the algorithm’s objective is to minimise sum of the distances between the points and their centroid. There are 5 steps to how K mean clustering works. Step 1, define the number of clusters. Step 2, random selection of centroid for each cluster. Step 3, assign each point to the closet cluster centroid. Step 4, compute centroids of new clusters. Step 5, repeat step 3 and 4. The algorithm will stop when the centroids within newly formed clusters remain the same, points don’t change within a cluster and maximum number of iterations are achieved (Pulkit Sharma, 2019).

2 clusters were chosen for the data clustering because the target variable is binary of 0 and 1. 9222 data samples have been assigned to 1 cluster and 10161 assigned to the other cluster with the K mean model.

Similarities between the prototype for each cluster: earning less than 50,000 USD per week, work in private section and is of white race. The differences are: first prototype is female in late 20’s who’s a high-school graduate working 32 hours per week, she also has children. The second prototype is a male in his mid 50’s working 45 hours per week who has some college degree but no children. The prototype did not distinguish between those who earn more than 50,000 to those who earn less.

The accuracy based on the testing set is only 69% which is about 10% less than performance of the optimised lr and SVM models (78.9% and 79.4 respectively). This has proven that the lr and SVM models are better choice for this dataset. This dataset has clearly defined labels so using a supervised machine learning algorithm such as lr and SVM yield better results.

# References:

1) Geeksforgeeks, Convert A Categorical Variable Into Dummy Variables, 2020, retrieved from: https://www.geeksforgeeks.org/convert-a-categorical-variable-into-dummy-variables/

2) Ayush Pant, Introduction to logistic regression, 2019, retrieved from: https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148

3) Rohith Gandhi, Support Vector Machine – introduction to machine learning algorithms, retrieved from: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

4) Anuja Nagpal, L1 and L2 regularisation methods, 2017, retrieved from: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

5) Scikit learn, logistic regression, retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

6) Velocity, SVM hyperparameter tuning using GridSearchCV, 2020, retrieved from: https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/

7) Scikit learn, SVM , retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

8) Pulkit Sharma, the most comprehensive guide to K-Means clustering you’ll ever need, 2019, retrieved from: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/

