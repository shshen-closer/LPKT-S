import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split

#df_train = pd.read_csv('train_data/train_task_1_2.csv')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)



    
all_data =  pd.read_csv('data/2012-2013-data-with-predictions-4-final.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
all_data['timestamp'] =  all_data['start_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
all_data['answer_time'] =  all_data['end_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S'))) - all_data['start_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
order = ['user_id','problem_id','correct','skill_id', 'timestamp', 'answer_time']
all_data = all_data[order]
all_data['skill_id'].fillna('nan',inplace=True)
all_data = all_data[all_data['skill_id'] != 'nan'].reset_index(drop=True)

print(all_data.isnull().sum())
print(all_data.info())
skill_id = np.array(all_data['skill_id'])

skills = set(skill_id)
print('skills:',  len(skills))


user_id = np.array(all_data['user_id'])
problem_id = np.array(all_data['problem_id'])

at_id = np.array(all_data['answer_time'])
at_id = [int(x) for x in at_id]
np.save('data/at_id.npy', np.array(at_id))  

user = set(user_id)
problem = set(problem_id)
at = set(at_id)

print('students, exercise, answer time:',  len(user), len(problem), len(at))


user2id ={}
problem2id = {}
at2id = {}
skill2id = {}

count = 1
for i in user:
    user2id[i] = count 
    count += 1
count = 1
for i in problem:
    problem2id[i] = count 
    count += 1
count = 1
for i in at:
    at2id[i] = count 
    count += 1
count = 0
for i in skills:
    skill2id[i] = count 
    count += 1
with open('data/at2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(at2id))
with open('data/user2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(user2id))
with open('data/problem2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(problem2id))
with open('data/skill2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(skill2id))
it_id = []
length = []
for item in tqdm(user):

    idx = all_data[(all_data.user_id==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    #temp1['IsCorrect'].fillna(2,inplace=True)
    
    temp = np.array(temp1)
    length.append(len(temp))
    if len(temp) < 2:
        continue

    for iii in range(1, len(temp)):
        a = (temp[iii][-2] - temp[iii-1][-2]) / 60 
        a = int(a)
        if a > 14400:
            a = 14400
        it_id.append(a)

#print('length:',  np.mean(length))
#np.save('data/length', np.array(length))
np.save('data/it_id.npy', np.array(it_id))  
print('its:',  len(it_id))
it = set(it_id) 
print('its:',  len(it))
it2id = {}

count = 1
for i in it:
    it2id[i] = count 
    count += 1
with open('data/it2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(it2id))

print('complete')
  



 