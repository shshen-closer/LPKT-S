# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import tensorflow as tf
from datetime import datetime 
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from model_lpkt_s import LPKTS
from utils import checkmate as cm
from utils import data_helpers as dh


TRAIN_OR_RESTORE = 'T' #input("Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()).replace(':', '_'))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()).replace(':', '_'))

number = str(sys.argv[1])
tf.compat.v1.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.compat.v1.flags.DEFINE_float("norm_ratio", 10, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 128, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 128 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 5, "Number of epochs to train for.")


tf.compat.v1.flags.DEFINE_integer("decay_steps", 1, "how many steps before decay learning rate. (default: 500)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate. (default: 0.95)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


def train():
    """Training model."""

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")

    logger.info("Training data processing...")
    train_students = np.load("data/train" + number + ".npy", allow_pickle=True)
    logger.info("Validation data processing...")
    valid_students = np.load("data/test" + number + ".npy", allow_pickle=True)
    

    
    print(np.shape(train_students))
    max_num_steps = 50
    max_num_skills = 265

    gama = 0.03

    print((len(train_students)//FLAGS.batch_size + 1) * FLAGS.decay_steps)
    # Build a graph and lstm_3 object
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            lpkts = LPKTS(
                batch_size = FLAGS.batch_size,
                num_steps = max_num_steps,
                num_skills = max_num_skills,
                hidden_size = FLAGS.hidden_size, 
                )
            

            # Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=lpkts.global_step, decay_steps=(len(train_students)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
               # learning_rate = tf.train.piecewise_constant(FLAGS.epochs, boundaries=[7,10], values=[0.005, 0.0005, 0.0001])
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
               # grads, vars = zip(*optimizer.compute_gradients(lpkts.loss))
                #grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                #train_op = optimizer.apply_gradients(zip(grads, vars), global_step=lpkts.global_step, name="train_op")
                train_op = optimizer.minimize(lpkts.loss, global_step=lpkts.global_step, name="train_op")

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.compat.v1.summary.scalar("loss", lpkts.loss)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=1, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load lpkts model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())



            current_step = sess.run(lpkts.global_step)

            def train_step(input_student, input_problem, input_kc, input_at, input_it, x_answer, target_id, target_kc, target_index, target_correctness):
                """A single training step"""
                
                feed_dict = {
                    lpkts.student_id: input_student,
                    lpkts.input_problem: input_problem,
                    lpkts.input_kc: input_kc,
                    lpkts.input_at: input_at,
                    lpkts.input_it: input_it,
                    lpkts.x_answer: x_answer,
                    lpkts.target_id: target_id,
                    lpkts.target_kc: target_kc,
                    lpkts.target_index: target_index,
                    lpkts.target_correctness: target_correctness,
                    lpkts.dropout_keep_prob: FLAGS.keep_prob,
                    lpkts.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, lpkts.global_step, train_summary_op, lpkts.pred, lpkts.loss], feed_dict)

                
                logger.info("step {0}: loss {1:g} ".format(step,loss))
                train_summary_writer.add_summary(summaries, step)
                return pred

            def validation_step(input_student, input_problem, input_kc, input_at, input_it, x_answer, target_id, target_kc, target_index, target_correctness):
                """Evaluates model on a validation set"""

                feed_dict = {
                    lpkts.student_id: input_student,
                    lpkts.input_problem: input_problem,
                    lpkts.input_kc: input_kc,
                    lpkts.input_at: input_at,
                    lpkts.input_it: input_it,
                    lpkts.x_answer: x_answer,
                    lpkts.target_id: target_id,
                    lpkts.target_kc: target_kc,
                    lpkts.target_index: target_index,
                    lpkts.target_correctness: target_correctness,
                    lpkts.dropout_keep_prob: 0.0,
                    lpkts.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [lpkts.global_step, validation_summary_op, lpkts.pred, lpkts.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                
                return pred
            # Training loop. For each batch...
            
            run_time = []

            for iii in range(FLAGS.epochs):
                np.random.seed(iii*100)
                np.random.shuffle(train_students)
                a=datetime.now()
                data_size = len(train_students)
                index = 0
                actual_labels = []
                pred_labels = []
                while(index+FLAGS.batch_size <= data_size):
                    input_student = np.zeros((FLAGS.batch_size))
                    input_problem = np.zeros((FLAGS.batch_size, max_num_steps))
                    input_kc = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * gama
                    input_at = np.zeros((FLAGS.batch_size, max_num_steps))
                    input_it = np.zeros((FLAGS.batch_size, max_num_steps))
                    x_answer = np.zeros((FLAGS.batch_size, max_num_steps))
                    target_id = np.zeros((FLAGS.batch_size, max_num_steps))
                    target_kc = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * gama
                    target_correctness = []
                    target_index = []
                    for i in range(FLAGS.batch_size):
                        student = train_students[index+i]
                        answer_times = student[0]
                        interval_times = student[1]
                        problem_ids = student[2]
                        correctness = student[3]
                        problem_kcs = student[4]
                        len_seq = student[5]
                        ss = student[6]
                        input_student[i] = ss
                       # print(correctness)
                        for j in range(len_seq-1):
                            input_problem[i,j] = problem_ids[j]
                            
                            input_kc[i, j, int(problem_kcs[j])] = 1.03
                            input_at[i,j] = answer_times[j]
                            input_it[i,j] = interval_times[j]
                            x_answer[i,j] = correctness[j]

                            target_id[i,j] = problem_ids[j + 1]
                            target_kc[i, j, int(problem_kcs[j+1])] = 1.03
                            target_index.append(i*max_num_steps+j)
                            target_correctness.append(int(correctness[j+1]))
                            actual_labels.append(int(correctness[j+1]))

                    index += FLAGS.batch_size
                    
                    pred = train_step(input_student, input_problem, input_kc, input_at, input_it, x_answer, target_id, target_kc, target_index, target_correctness)
                    for p in pred:
                        pred_labels.append(p)
                    current_step = tf.compat.v1.train.global_step(sess, lpkts.global_step)
                
                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)
                rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                auc = metrics.roc_auc_score(actual_labels, pred_labels)
                #calculate r^2
                r2 = r2_score(actual_labels, pred_labels)
                
                pred_score = np.greater_equal(pred_labels,0.5) 
                pred_score = pred_score.astype(int)
                pred_score = np.equal(actual_labels, pred_score)
                acc = np.mean(pred_score.astype(int))

                logger.info("epochs {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g}".format((iii +1),rmse, auc, r2, acc))

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")
                    
                    data_size = len(valid_students)
                    index = 0
                    actual_labels = []
                    pred_labels = []
                    while(index+FLAGS.batch_size <= data_size):
                        input_student = np.zeros((FLAGS.batch_size))
                        input_problem = np.zeros((FLAGS.batch_size, max_num_steps))
                        input_kc = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * gama
                        input_at = np.zeros((FLAGS.batch_size, max_num_steps))
                        input_it = np.zeros((FLAGS.batch_size, max_num_steps))
                        x_answer = np.zeros((FLAGS.batch_size, max_num_steps))
                        target_id = np.zeros((FLAGS.batch_size, max_num_steps))
                        target_kc = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * gama
                        target_correctness = []
                        target_index = []
                        for i in range(FLAGS.batch_size):
                            student = valid_students[index+i]
                            answer_times = student[0]
                            interval_times = student[1]
                            problem_ids = student[2]
                            correctness = student[3]
                            problem_kcs = student[4]
                            len_seq = student[5]
                            ss =  student[6]
                            input_student[i] = ss
                            for j in range(len_seq-1):
                                
                                input_problem[i,j] = problem_ids[j]
                                input_kc[i, j, int(problem_kcs[j])] = 1.03
                                input_at[i,j] = answer_times[j]
                                input_it[i,j] = interval_times[j]
                                x_answer[i,j] = correctness[j]

                                target_id[i,j] = problem_ids[j + 1]
                                target_kc[i, j, int(problem_kcs[j+1])] = 1.03
                                target_index.append(i*max_num_steps+j)
                                target_correctness.append(int(correctness[j+1]))
                                actual_labels.append(int(correctness[j+1]))
                        index += FLAGS.batch_size
                        pred  = validation_step(input_student, input_problem, input_kc, input_at, input_it, x_answer, target_id, target_kc, target_index, target_correctness)
                        for p in pred:
                            pred_labels.append(p)
                    

                    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                    auc = metrics.roc_auc_score(actual_labels, pred_labels)
                    #calculate r^2
                    r2 = r2_score(actual_labels, pred_labels)
                    
                    pred_score = np.greater_equal(pred_labels,0.5) 
                    pred_score = pred_score.astype(int)
                    pred_score = np.equal(actual_labels, pred_score)
                    acc = np.mean(pred_score.astype(int))

                    logger.info("VALIDATION {0}: rmse {1:g}  auc {2:g}  r2 {3:g}  acc {4:g} ".format((iii +1)/FLAGS.evaluation_interval,rmse, auc, r2, acc))
                    
                    best_saver.handle(auc, sess, current_step)

                

                if ((iii+1) % FLAGS.checkpoint_every == 0):
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))

                logger.info("Epoch {0} has finished!".format(iii + 1))
            
            logger.info("running time analysis: epoch{0}, avg_time{1}".format(len(run_time), np.mean(run_time)))

    logger.info("Done.")


if __name__ == '__main__':
    train()
