# Project Description
This project is the final competition of the course [Introduction to Data Science and Machine Learning](https://stepik.org/course/4852) on the edu platform named Stepic.
It is known, due to assignment, that madel performance will be estimated by ROC_AUC sore metric .  
As part of this competition, 2 solutions have been developed so far, which are located in the *notebooks_(lang)* folders:  

1. *baseline_with_full_data_processing* - baseline with features that were created during the course. 
Also you can find step-by-step preprocessing of the data with feature analysis in the notebook. Comparative characteristics and general analytics are also provided to select the best model.  
**ROC_AUC score of this model = 0.804**.    

2. *best_performance_with_new_features* - solution that expands the features set from baseline. Added a score of each step difficulty according to different criteria. 
Also, for the sake of brevity, these notebooks include packages with functions that automatically (according to the algorithm from the baseline) preprocess the data
and train the model.    
**ROC_AUC score of this model = 0.822**

## The essence of the task  
**Predict dropout of user after first 2 days activity on [course](https://stepik.org/course/129/syllabus)**
  
## Project structure  
1. **data** - data directory for feature generation  
    - *ROC_SCOR* - the directory with the screenshots where the Stepic system checks the accuracy of the model using the ROC_AUC metric  
    - *test* - catalog with data for forecasting  
      - *events_data_test.zip* - actions of users with steps  
      - *submission_data_test.zip* - time and statuses of users' submissions with practical steps  
    - *train* - directory with data for training  
      - *event_data_train.zip* - users actions with steps  
      - *submissions_data_train.zip* - time and statuses of submissions of users with practice steps  
      
2. **notebooks_eng** - English notebooks catalog with solutions  
    - *baseline_with_full_data_processing[ENG].ipynb* - detailed analysis  
    - *best_performance_with_new_features[ENG].ipynb* - solution that extends the set of features from the baseline. Added an evaluation of the difficulty of each step by different criteria. 
    
3. **notebooks_rus** - directory of notebooks in Russian with solutions for this problem  
    - *baseline_with_full_data_processing[RUS]. -   detailed analysis
    - *best_performance_with_new_features[RUS].ipynb* - solution that extends the set of traits from the baseline. Added an evaluation of the difficulty of each step by different criteria. 
    
4. **reports** - directory with prediction reports of probability of assigning user to dropout class  
    - *reports.zip* - archive containing such predictions. For each such prediction, the name of the file specifies which model predicted it  

5. **utils** - folder with code for generating datasets, features and model training with making performance comparison tables  
    - *model_fit_utils.py* - functions that simplify the process of training models. There are also functions for making comparison tables  
    - *preprocessing_utils.py* - functions that simplify the process of forming datasets and creating the necessary features.
        There is also a function that returns X and y from  events_data and submission_data gave as parameters  
## Data description  
 **events_train.csv** - data about the students actions with steps

1. **step_id** - step id  
2. **user_id** - anonymized user id  
3. **timestamp** - event time in unix date format  
4. **action** - event, possible values:  
*discovered* - user went to the step  
*viewed* - step preview,  
*started_attempt* - start attempting to solve a step
*passed* - successful solution of the practice step  
  
**submissions_train.csv** - data on time and statuses of submissions to practical tasks

1. **step_id** - step id
2. **timestamp** - time of solution sending in unix date format
3. **submission_status** - decision status
4. **user_id** - anonymized user id
