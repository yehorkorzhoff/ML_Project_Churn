# Описание проекта(English version given below)
Данный проект является финальным соревнованием курса [Введение в Data Science и машинное обучение](https://stepik.org/course/4852) на платформе Stepic.
Известно, исходя из постановки задания соревнования, что за метрику качества модели будет взят ROC_AUC sore.  
В рамках этого соревнования на данный момент разработано 2 решения, которые находятся в папке *notebooks_(lang):*  

1. *baseline_with_full_data_processing* - baseline по признакам, которые были созданы в течении прохождения курса. 
Также в ноутбуке присутствует поэтапная предобработка данных с анализом признаков. Для выбора оптимальной модели также предоставлены сравнительные характеристики и общая аналитика.  
**ROC_AUC score данной модели = 0.804**    

2. *best_performance_with_new_features* - решение, которое расширяет набор признаков из baseline. Добавлено оценку сложности каждого стэпа по разным критериям. 
Также в данных ноутбуках для краткости изложения предусмотрено подключения пакетов с функциями, которые автоматически(по алгоритму из baseline) предобрабатывают данные
и обучают модели.    
**ROC_AUC score данной модели = 0.822**

## Суть задания  
**по прошествии 2 дней с момента первой активности пользователя на [курсе](https://stepik.org/course/129/syllabus) определить закончит ли он его**
  
## Структура проекта  
1. **data** - каталог данных для построения признаков  
    - *ROC_SCOR* - каталог со скринами, на которых система Stepic проверяет точность модели по метрике ROC_AUC  
    - *test* - каталог с данными для прогноза  
      - *events_data_test.zip* - действия пользователей со стэпами  
      - *submission_data_test.zip* - время и статусы сабмитов пользователей с практическими стэпами  
    - *train* - каталог с данными для обучения  
      - *event_data_train.zip* - действия пользователей со стэпами  
      - *submissions_data_train.zip* - время и статусы сабмитов пользователей с практическими стэпами  
      
2. **notebooks_eng** - каталог ноутбуков на английском языке с решениями данной проблемы  
    - *baseline_with_full_data_processing[ENG].ipynb* - baseline по признакам, которые были созданы в течении прохождения курса с подробным анализом  
    - *best_performance_with_new_features[ENG].ipynb* - решение, которое расширяет набор признаков из baseline. Добавлено оценку сложности каждого стэпа по разным критериям. 
    
3. **notebooks_rus** - каталог ноутбуков на русском языке с решениями данной проблемы  
    - *baseline_with_full_data_processing[RUS].ipynb* - baseline по признакам, которые были созданы в течении прохождения курса с подробным анализом  
    - *best_performance_with_new_features[RUS].ipynb* - решение, которое расширяет набор признаков из baseline. Добавлено оценку сложности каждого стэпа по разным критериям. 
    
4. **reports** - каталог с отчетами по предсказаниям вероятности отнесения пользователя к классу dropout student  
    - *reports.zip* - архив, в котором находятся такие предсказания. Для каждого такого прогноза в названии файла указана какая модель это предсказала  

5. **utils** - папка с кодом для генерации датасетов, признаков и обучения моделей с составлением сравнительных таблиц производительности  
    - *model_fit_utils.py* - функции, которые упрощают процесс обучения моделей. Также присутствуют функции по составлению сравнительных таблиц  
    - *preprocessing_utils.py* - функции, которые упрощают процесс формирования датасетов и создания нужных признаков.
        Также присутствует функция, которая возвращает X и y по переданным events_data и submission_data  
## Описание данных  
 **events_train.csv** - данные о действиях, которые совершают студенты со стэпами

1. **step_id** - id стэпа  
2. **user_id** - анонимизированный id юзера  
3. **timestamp** - время наступления события в формате unix date  
4. **action** - событие, возможные значения:  
*discovered* - пользователь перешел на стэп  
*viewed* - просмотр шага,  
*started_attempt* - начало попытки решить шаг, ранее нужно было явно нажать на кнопку - начать решение, перед тем как приступить к решению практического шага  
*passed* - удачное решение практического шага  
  
**submissions_train.csv** - данные о времени и статусах сабмитов к практическим заданиям

1. **step_id** - id стэпа
2. **timestamp** - время отправки решения в формате unix date
3. **submission_status** - статус решения
4. **user_id** - анонимизированный id юзера

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
