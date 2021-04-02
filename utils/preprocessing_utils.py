import pandas as pd
import numpy as np

def preprocess_events_data(e_d):
    # создаем колонки , в которых будем хранить дату-и-время и день-месяц-год
    
    e_d['date'] = pd.to_datetime(e_d['timestamp'], unit='s')
    e_d['day'] = e_d['date'].dt.date
    return e_d


def stat_events_data(e_d):
    # создадим df, в котором будем хранить подробную статиситку по action для каждого пользователя
    
    return e_d.pivot_table(index='user_id', columns='action', values='step_id',
                       aggfunc='count', fill_value=0).reset_index()
  
    
def step_rate_events_data(e_d):
    # создает df, в котором будем хранить для каждого степа его "сложность". Возвращает events_data с новыми колонками
    
    event_steps = e_d.pivot_table(index='step_id', columns='action', values='user_id',
                       aggfunc='count', fill_value=0).reset_index()
    event_steps['passed / discovered'] = event_steps['passed'] / event_steps['discovered']
    event_steps['passed / started_attempt'] = event_steps['passed'] / event_steps['started_attempt']
    event_steps['passed / viewed'] = event_steps['passed'] / event_steps['viewed']
    event_steps = event_steps.replace([np.inf, -np.inf], np.nan)
    event_steps = event_steps.fillna(1)
    return e_d.merge(event_steps[['step_id',
                               'passed / discovered',
                               'passed / started_attempt', 'passed / viewed']], on='step_id', how='outer')


def preprocess_submissions_data(s_d):
    # создаем колонки , в которых будем хранить дату-и-время и день-месяц-год
    
    s_d['date'] = pd.to_datetime(s_d['timestamp'], unit='s')
    s_d['day'] = s_d['date'].dt.date
    return s_d


def stat_submissions_data(s_d):
    # создадим df, в котором будем хранить подробную статиситку по submission_status для каждого пользователя
    
    return s_d.pivot_table(index='user_id', columns='submission_status', values='step_id',
                       aggfunc='count', fill_value=0).reset_index()


def step_rate_submissions_data(s_d):
    # создает df, в котором будем хранить для каждого степа его "сложность". Возвращает submissions_data с новыми колонками
    
    submissions_steps = s_d.pivot_table(index='step_id', columns='submission_status', values='user_id',
                       aggfunc='count', fill_value=0).reset_index()
    submissions_steps['step_correct_ratio'] = submissions_steps['correct'] / (submissions_steps['correct'] + submissions_steps['wrong'])
    return s_d.merge(submissions_steps[['step_id', 'step_correct_ratio']], on='step_id', how='outer')


def get_diff_between_unique_days(e_d):
    # для понимания медиальной картины активности пользователей
    # уникальные дни пользователя
    un_days = e_d[['user_id',
             'day', 
             'timestamp']].drop_duplicates(subset=['user_id', 'day']) \
    .groupby('user_id')['timestamp'].apply(list)
    
    # разница между уникальными днями
    gap_data = un_days.apply(np.diff).values
    gap_data = pd.Series(np.concatenate(gap_data, axis=0))
    return gap_data / (24 * 60 * 60)


def mark_dropped_out_users_with_new_features(e_d, drop_out_threshold):
    # делаем предположение, что пользователи, которые не появлялись больше drop_out_threshold явл. ушедшими
    
    # ADDED MEAN FOR SCORE
    now = e_d['timestamp'].max()
    users_data = e_d.groupby('user_id',
                            as_index=False).agg({'timestamp': 'max', 
                                                'passed / discovered': 'mean',
                                                'passed / started_attempt': 'mean',
                                                'passed / viewed': 'mean'}) \
        .rename(columns={'timestamp': 'last_timestamp'})
    users_data['is_gone_user'] = (now - users_data['last_timestamp']) > drop_out_threshold
    return users_data


def mark_dropped_out_users(e_d, drop_out_threshold):
    # делаем предположение, что пользователи, которые не появлялись больше drop_out_threshold явл. ушедшими
    
    now = e_d['timestamp'].max()
    users_data = e_d.groupby('user_id',
                            as_index=False).agg({'timestamp': 'max'}) \
        .rename(columns={'timestamp': 'last_timestamp'})
    users_data['is_gone_user'] = (now - users_data['last_timestamp']) > drop_out_threshold
    return users_data


def merge_all(u_d, u_s, u_e_d, e_d):
    # соеденим все таботцы в одну
    
    u_d = u_d.merge(u_s,on='user_id', how='outer')
    u_d = u_d.fillna(0)
    u_d = u_d.merge(u_e_d, on='user_id', how='outer')
    
    users_days = e_d.groupby('user_id')[['day']].nunique().reset_index()
    u_d = u_d.merge(users_days, on='user_id', how='outer')
    return u_d


def mark_passed_course_users(u_d, grade2pass):
    # считаем, что пользователь, который набрал больше grade2pass прошел курс
    
    u_d['passed_course'] = u_d['passed'] > grade2pass
    return u_d


def get_events_learning_time_threshold(u_d, e_d, learning_time_threshold):
    # активность пользователс за первые learning_time_threshold дней
    
    # первая активность пользователя на курсе
    user_min_time = e_d.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'min'}).rename({'timestamp': 'min_timestamp'}, axis=1)
    
    u_d = u_d.merge(user_min_time, on='user_id', how='outer')
    
    e_d['user_time'] = e_d['user_id'].map(str) + '_' + e_d['timestamp'].map(str)
    
    user_learning_time_threshold = user_min_time['user_id'].map(str) + '_' \
    + (user_min_time['min_timestamp'] + learning_time_threshold).map(str)
    
    user_min_time['user_learning_time_threshold'] = user_learning_time_threshold
    
    e_d = e_d.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')
    return e_d[e_d['user_time'] <= e_d['user_learning_time_threshold']], user_min_time
    
    
def get_submissions_learning_time_threshold(s_d, us_min_t, learning_time_threshold):
    
    s_d['user_time'] = s_d['user_id'].map(str) \
    + '_' + s_d['timestamp'].map(str)

    s_d = s_d \
    .merge(us_min_t[['user_id', 'user_learning_time_threshold']], how='outer')

    return s_d[s_d['user_time'] <= s_d['user_learning_time_threshold']]


def create_X_y_with_new_features(s_d_t, e_d_t, u_d):
    
    X = s_d_t.groupby('user_id')['day'].nunique().to_frame().reset_index().rename(columns={'day': 'days'})
    # степы, которые пользователь пытался решить за первые 3 дня
    steps_tried = s_d_t.groupby('user_id')['step_id'].nunique() \
        .to_frame().reset_index().rename(columns={'step_id': 'steps_tried'})
    X = X.merge(steps_tried, on='user_id', how='outer')
    
    
    X = X.merge(s_d_t.pivot_table(index='user_id', columns='submission_status', values='step_id',
                       aggfunc='count', fill_value=0).reset_index())
    X['correct_ratio'] = X['correct'] / (X['correct'] + X['wrong'])
    # ADDED MERGE WITH Score
    X = X.merge(s_d_t.groupby('user_id').agg({'step_correct_ratio': 'mean'}), on='user_id', how='outer')
    
    
    
    X = X.merge(e_d_t.pivot_table(index='user_id', columns='action', values='step_id',
                       aggfunc='count', fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')
    # ADDED MERGE WITH SCORE
    
    
    X = X.merge(e_d_t.groupby('user_id').agg({'passed / discovered': 'mean',
                                             'passed / started_attempt': 'mean',
                                             'passed / viewed': 'mean'}), on='user_id', how='outer')
    
    X = X.fillna(0)
    X = X.merge(u_d[['user_id', 'passed_course', 'is_gone_user']], how='outer')
    
    X = X[~((X['is_gone_user'] == False) & (X['passed_course'] == False))]

    y = X['passed_course'].map(int)
    X = X.drop(['passed_course', 'is_gone_user'], axis=1)
    
    X = X.set_index(X['user_id'])
    X = X.drop('user_id', axis=1)
    
    return X, y


def create_X_y(s_d_t, e_d_t, u_d):
    
    X = s_d_t.groupby('user_id')['day'].nunique().to_frame().reset_index().rename(columns={'day': 'days'})
    # степы, которые пользователь пытался решить за первые 3 дня
    steps_tried = s_d_t.groupby('user_id')['step_id'].nunique() \
        .to_frame().reset_index().rename(columns={'step_id': 'steps_tried'})
    X = X.merge(steps_tried, on='user_id', how='outer')
    
    
    X = X.merge(s_d_t.pivot_table(index='user_id', columns='submission_status', values='step_id',
                       aggfunc='count', fill_value=0).reset_index())
    X['correct_ratio'] = X['correct'] / (X['correct'] + X['wrong'])
    X = X.merge(e_d_t.pivot_table(index='user_id', columns='action', values='step_id',
                       aggfunc='count', fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')
    
    X = X.fillna(0)
    X = X.merge(u_d[['user_id', 'passed_course', 'is_gone_user']], how='outer')
    
    X = X[~((X['is_gone_user'] == False) & (X['passed_course'] == False))]

    y = X['passed_course'].map(int)
    X = X.drop(['passed_course', 'is_gone_user'], axis=1)
    
    X = X.set_index(X['user_id'])
    X = X.drop('user_id', axis=1)
    
    return X, y


def make_X_y(e_d, s_d):
    e_d = preprocess_events_data(e_d)
    u_e_d = stat_events_data(e_d)
    event_steps = step_rate_events_data(e_d)
    
    s_d = preprocess_submissions_data(s_d)
    u_s = stat_submissions_data(s_d)
    submissions_steps = step_rate_submissions_data(s_d)
    
    gap_data = get_diff_between_unique_days(e_d)
    drop_out_threshold = 30 * 24 * 60 * 60
    u_d = mark_dropped_out_users(e_d, drop_out_threshold)
    
    u_d = merge_all(u_d, u_s, u_e_d, e_d)
    grade2pass = 170 
    u_d = mark_passed_course_users(u_d, grade2pass)
    
    learning_time_threshold = 3 * 24 * 60 * 60
    e_d_t, us_min_t = get_events_learning_time_threshold(u_d, e_d, learning_time_threshold)
    s_d_t = get_submissions_learning_time_threshold(s_d, us_min_t, learning_time_threshold)
    
    return create_X_y(s_d_t, e_d_t, u_d)


def make_X_y_with_new_fetures(e_d, s_d):
    e_d = preprocess_events_data(e_d)
    u_e_d = stat_events_data(e_d)
    e_d = step_rate_events_data(e_d)
    
    s_d = preprocess_submissions_data(s_d)
    u_s = stat_submissions_data(s_d)
    s_d = step_rate_submissions_data(s_d)
    
    gap_data = get_diff_between_unique_days(e_d)
    drop_out_threshold = 30 * 24 * 60 * 60
    u_d = mark_dropped_out_users_with_new_features(e_d, drop_out_threshold)
    
    step_correct_ratio = s_d.groupby('user_id').agg({'step_correct_ratio': 'mean'})
    u_s = u_s.merge(step_correct_ratio, on='user_id', how='outer')
    u_d = merge_all(u_d, u_s, u_e_d, e_d)
    grade2pass = 170 
    u_d = mark_passed_course_users(u_d, grade2pass)
    
    learning_time_threshold = 3 * 24 * 60 * 60
    e_d_t, us_min_t = get_events_learning_time_threshold(u_d, e_d, learning_time_threshold)
    s_d_t = get_submissions_learning_time_threshold(s_d, us_min_t, learning_time_threshold)
    
    return create_X_y_with_new_features(s_d_t, e_d_t, u_d)


def write_to_csv(x, y, file_name):
    df = pd.DataFrame(x)
    df['is_gone'] = y
    df = df.set_index('user_id')
    df.to_csv(file_name)