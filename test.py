# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from datetime import datetime 
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import checkmate as cm
from utils import data_helpers as dh
import json

# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))
file_name = sys.argv[1]


MODEL = file_name
while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(90175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")



MODEL_DIR =  'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR =  'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 256, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 128 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("seq_len", 50, "Number of epochs to train for.")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
#logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
   #                             for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))
#


    

def test():

    # Load data
    logger.info("Loading data...")

    
    logger.info("test data processing...")
    test_students = np.load("data/test.npy", allow_pickle=True)
    test_students = test_students[:400]
    print(np.shape(test_students))
    max_num_steps = 50
    max_num_skills = 265

    BEST_OR_LATEST = 'B'

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    if BEST_OR_LATEST == 'L':
        logger.info("latest")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            student_id = graph.get_operation_by_name("student_id").outputs[0]
            input_problem = graph.get_operation_by_name("input_problem").outputs[0]
            input_kc = graph.get_operation_by_name("input_kc").outputs[0]
            input_at = graph.get_operation_by_name("input_at").outputs[0]
            input_it = graph.get_operation_by_name("input_it").outputs[0]
            x_answer = graph.get_operation_by_name("x_answer").outputs[0]
            target_id = graph.get_operation_by_name("target_id").outputs[0]
            target_kc = graph.get_operation_by_name("target_kc").outputs[0]
            target_index = graph.get_operation_by_name("target_index").outputs[0]
            target_correctness = graph.get_operation_by_name("target_correctness").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            pred = graph.get_operation_by_name("pred").outputs[0]
            o_knowledge = graph.get_operation_by_name("learning_progress").outputs[0]
            state = graph.get_operation_by_name("knowledge_state").outputs[0]
            
        
            a=datetime.now()
            data_size = len(test_students)
            index = 0
            actual_labels = []
            pred_labels = []
            ids = []
            pbs = []
            kcs= []
            its = []
            ats = []
            output = []
            output_state = []
            leng = []
            while(index+FLAGS.batch_size <= data_size):
                input_student_b = np.zeros((FLAGS.batch_size))
                input_problem_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_kc_b = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * 0.03
                input_at_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_it_b = np.zeros((FLAGS.batch_size, max_num_steps))
                x_answer_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_id_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_kc_b = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * 0.03
                target_correctness_b = []
                target_index_b = []
                for i in range(FLAGS.batch_size):
                    student = test_students[index+i]
                    answer_times = student[0]
                    interval_times = student[1]
                    problem_ids = student[2]
                    correctness = student[3]
                    problem_kcs = student[4]
                    len_seq = student[5]
                    ss =  student[6]
                    input_student_b[i] = ss
                   # print(np.shape(correctness))
                    ids.append(problem_ids)
                    pbs.append(correctness)
                    kcs.append(problem_kcs)
                    its.append(interval_times)
                    ats.append(answer_times)
                    leng.append(len_seq)
                    for j in range(len_seq-1):
                        input_problem_b[i,j] = problem_ids[j]
                        input_kc_b[i, j, int(problem_kcs[j])] = 1.03
                        input_at_b[i,j] = answer_times[j]
                        input_it_b[i,j] = interval_times[j]
                        x_answer_b[i,j] = correctness[j]

                        target_id_b[i,j] = problem_ids[j + 1]
                        target_kc_b[i, j, int(problem_kcs[j+1])] = 1.03
                        target_index_b.append(i*max_num_steps+j)
                        target_correctness_b.append(int(correctness[j+1]))
                        actual_labels.append(int(correctness[j+1]))

                index += FLAGS.batch_size


                feed_dict = {
                    student_id : input_student_b,
                    input_problem: input_problem_b,
                    input_kc:input_kc_b,
                    input_at: input_at_b,
                    input_it:input_it_b,
                    x_answer: x_answer_b,
                    target_id: target_id_b,
                    target_kc: target_kc_b,
                    target_index: target_index_b,
                    target_correctness: target_correctness_b,
                    dropout_keep_prob: 0.0,
                    is_training: False
                }
                
                o_knowledge_b = sess.run(o_knowledge, feed_dict)
                o_state_b = sess.run(state, feed_dict)
                output.extend(o_knowledge_b.tolist())
                output_state.extend(o_state_b.tolist())
                pred_b = sess.run(pred, feed_dict)
                
                pred_labels.extend(pred_b.tolist())
            rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
            auc = metrics.roc_auc_score(actual_labels, pred_labels)
            #calculate r^2
            r2 = r2_score(actual_labels, pred_labels)
            
            pred_score = np.greater_equal(pred_labels,0.5) 
            pred_score = pred_score.astype(int)
            pred_score = np.equal(actual_labels, pred_score)
            acc = np.mean(pred_score.astype(int))
            print("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g}".format(0,rmse, auc, r2, acc))
            logger.info("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g}".format(0,rmse, auc, r2, acc))
        #    os.remove('output')
            print(np.shape(pred_labels))
            print(len(leng))
            fo =  open('output', 'a') 
            for ll in range(len(leng)):
                len_l = leng[ll]
                o_l = output[ll]
                o_l = np.array(o_l)
                
                pb_l = pbs[ll]
                id_l = ids[ll]
                kc_l = kcs[ll]
                s_l = output_state[ll]
               # print(np.shape(o_l))
                set_kc = list(set(kc_l))
                set_kc = sorted(set_kc)
                #print(set_kc)
                it_l = its[ll]
                at_l = ats[ll]
                fo.write(str(len_l) + '\n')
                
                fo.write(str(pb_l) + '\n')
                fo.write(str(id_l) + '\n')
                fo.write(str(kc_l) + '\n')
                fo.write(str(at_l) + '\n')
                fo.write(str(it_l) + '\n')
                ppp = o_l[:len_l]
                ppp = [2/ (1 + np.exp(-0.03*x)) - 1 for x in ppp]   #可以吧-0.03换成其它值，更好地将知识状态映射到0-1之间进行可视化
                print(ppp)
                for iii in ppp:
                    fo.write(str(round(iii,2)) + '\t')
                fo.write('\n')

                ppp = s_l[:len_l]
                for jjj,iii in zip(kc_l,ppp):
                    ppt = 2/ (1 + np.exp(-0.03*iii[jjj])) - 1 

                    fo.write(str(round(ppt,2)) + '\t')
                fo.write('\n')
                    
            fo.close()


    logger.info("Done.")


if __name__ == '__main__':
    test()
