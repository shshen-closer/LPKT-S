import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import sys
import time, datetime
from sklearn.model_selection import train_test_split, KFold

#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)
kfold = KFold(n_splits=5, shuffle=False)

max_len = int(sys.argv[1])

    
all_data =  pd.read_csv('data/2012-2013-data-with-predictions-4-final.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
all_data['timestamp'] =  all_data['start_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
all_data['answer_time'] =  all_data['end_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S'))) - all_data['start_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
order = ['user_id','problem_id','correct','skill_id', 'timestamp', 'answer_time']
all_data = all_data[order]
all_data['skill_id'].fillna('nan',inplace=True)
all_data = all_data[all_data['skill_id'] != 'nan'].reset_index(drop=True)


with open('data/at2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        at2id = eval(line)
with open('data/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/it2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        it2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)
user_id = np.array(all_data['user_id'])
user = list(set(user_id))

q_a_all = []
length = []
for item in tqdm(user):
    idx = all_data[(all_data.user_id==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue

    segs = [0]
    for iii in range(1,len(temp)):
        a = int((temp[iii][4] - temp[iii-1][4])/60)
        if a > 14400:
            segs.append(iii)
    segs.append(len(temp))

    for iii in range(1, len(segs)):

        quiz_one = temp[segs[iii-1]:segs[iii]]
        if len(quiz_one) < 2:
            continue
        length.append(len(quiz_one))
        while len(quiz_one) >= 2:
            quiz = quiz_one[0:max_len]
            
            train_at = [at2id[int(quiz[0][5])]]
            train_it = [0]
            train_q = [problem2id[quiz[0][1]]]
            train_a = [int(quiz[0][2])]
            train_skill = [skill2id[quiz[0][3]]]

            for one in range(1,len(quiz)):
                train_at.append(at2id[int(quiz[one][5])])
                a = int((quiz[one][4] - quiz[one-1][4])/60)
                if a > 14400:
                    a = 14400
                    print('over_long')
                train_it.append(it2id[a])
                train_q.append(problem2id[quiz[one][1]])
                train_a.append(int(quiz[one][2]))
                train_skill.append(skill2id[quiz[one][3]])
            q_a_all.append([train_at, train_it, train_q, train_a, train_skill, len(quiz), user2id[item]])
            quiz_one = quiz_one[max_len:]

print('avg.leng: ', np.mean(length))
q_a_all = np.array(q_a_all)
np.random.seed(317)
np.random.shuffle(q_a_all)

train_all, q_a_test = train_test_split(q_a_all,test_size=0.2,shuffle=True, random_state = 317)
kfold = KFold(n_splits=5, shuffle=True, random_state = 317)
count = 0
for (train_index, valid_index) in kfold.split(train_all):
    #print(train_index[0:10])
    q_a_train = train_all[train_index]
    q_a_valid = train_all[valid_index]
    np.save("data/train" + str(count) + ".npy",np.array(q_a_train))
    np.save("data/test" + str(count) + ".npy",np.array(q_a_valid))
    count+=1
np.save("data/test.npy",np.array(q_a_test))

print('complete')
            



 