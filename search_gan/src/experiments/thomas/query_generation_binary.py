# coding: utf-8
import os
import sys
import logging
import numpy as np
import tensorflow as tf
import argparse

# Setting path
root_dir = "{}/../..".format(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from common.io_utils import CircularRecordIterator, parse_serialized_batch, nb_triplets_query_product_buckets  # noqa
from common.experiment_db import update_experiment  # noqa

# Training - validation files location
cwd = os.getcwd()
log_location = cwd + '/logs/'

# Parameters from grid
parser = argparse.ArgumentParser(
    description='Learn one classifier on each language, starting from a sparse representation of products')

# Automatic arguments
parser.add_argument('--run_id', type=int, default=0, metavar='N')
parser.add_argument('--metarun_id', type=int, default=0, metavar='N')
parser.add_argument('--max_nb_processes', type=int)
parser.add_argument('--gpu_id', type=int)
parser.add_argument('--experiment_name', type=str)

# Hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.00005, metavar='N')
parser.add_argument('--p_keep_for_dropout', type=float)
parser.add_argument('--catalog_id', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--learn_on_negatives', type=int)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[{}_{}_{}]".format(
    args.experiment_name,
    args.metarun_id,
    args.run_id))

logger.info("Updating experiment")
update_experiment(metarun_id=args.metarun_id, run_id=args.run_id,
                  metrics={"status": "started"})
logger.info("Updating done")

logger.info("Using GPU {}".format(args.gpu_id))
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


####################
# Helper functions #
####################

def data_generator(mode):
    iterator = CircularRecordIterator(args.catalog_id, mode)
    is_new_epoch = False
    while not is_new_epoch:
        serialized_data, is_new_epoch = iterator.get_next_serialized_batch()
        queries, products, labels = parse_serialized_batch(serialized_data)
        yield queries, products, labels


def flatten_and_reshape(input_x):
    x_flat = tf.reshape(input_x,
                        shape=[-1, nb_triplets_query_product_buckets],
                        name="{}_flat".format(input_x.name.split(":")[0]))
    return tf.concat([x_flat], 1)


def get_positives(q, p, y):
    p_pos = p[np.where(y == 1)[0], :, :]
    q_pos = q[np.where(y == 1)[0], :, :]

    return q_pos, p_pos


def get_negatives(q, p, y):
    p_neg = p[np.where(y == 0)[0], :, :]
    q_neg = q[np.where(y == 0)[0], :, :]

    return q_neg, p_neg


####################
# Graph definition #
####################

tf.reset_default_graph()


def simple_dense(inputx):
    hidden_layer = tf.layers.dense(
        inputx, 2 * 2 ** 13,
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.random_normal
    )
    output_layer = tf.layers.dense(
        hidden_layer, 2 ** 13,
        activation=None,
        kernel_initializer=tf.initializers.random_normal
    )
    return output_layer


with tf.device('/gpu:0'):
    product_raw = tf.placeholder(tf.float32,
                                 [None, nb_triplets_query_product_buckets, 1],
                                 name="product_raw")

    query_raw = tf.placeholder(tf.float32,
                               [None, nb_triplets_query_product_buckets, 1],
                               name="query_raw")

    y_task = tf.placeholder(tf.float32, shape=(None, 1))

    # Reshaping product and query
    product = flatten_and_reshape(product_raw)
    query = flatten_and_reshape(query_raw)

    summary_ops = list()

    # Variables that will be defined depending on args.method
    query_generated = None
    train_step = None

    if args.method == "dnn":
        query_generated = simple_dense(product)
        loss = tf.reduce_mean(
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=query, logits=query_generated), axis=0
            )
        )

        summary_ops.append(tf.summary.scalar("xentropy", loss))

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_step = optimizer.minimize(loss)
    else:
        raise RuntimeError("Unsupported method '{}'".format(args.method))

    summaries = tf.summary.merge(summary_ops)

    # AUC Operators for positive and negative

    auc_p, auc_update_op_p = tf.metrics.auc(labels=query, predictions=tf.nn.sigmoid(query_generated), name="pos")
    auc_n, auc_update_op_n = tf.metrics.auc(labels=query, predictions=tf.nn.sigmoid(query_generated), name="neg")

    summaries_auc = tf.summary.merge([
        tf.summary.scalar("auc_n", auc_n),
        tf.summary.scalar("auc_p", auc_p)
    ])

###########
# Session #
###########

if args.max_nb_processes > 1:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.9 / args.max_nb_processes)
else:
    gpu_options = tf.GPUOptions()

config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

sess = tf.Session(config=config)

log_location = log_location + args.experiment_name


def get_writer(mode):
    return tf.summary.FileWriter(
        log_location + '/{}_{}_{}'.format(args.metarun_id, args.run_id,
                                          mode),
        sess.graph)


train_writer = get_writer("train")
test_writer = get_writer("test")

ini_l = tf.local_variables_initializer()
ini_g = tf.global_variables_initializer()
sess.run(ini_l)
sess.run(ini_g)

step = 0

for q, p, y in data_generator("train"):
    step += 1

    q_pos, p_pos = get_positives(q, p, y)
    q_neg, p_neg = get_negatives(q, p, y)

    feed_dict_pos = {
        query_raw: q_pos,
        product_raw: p_pos
    }

    feed_dict_neg = {
        query_raw: q_neg,
        product_raw: p_neg
    }

    if args.learn_on_negatives == 0:
        sess.run(train_step, feed_dict=feed_dict_pos)
    else:
        sess.run(train_step, feed_dict=feed_dict_neg)

    val_summaries = sess.run(summaries, feed_dict=feed_dict_pos)

    sess.run(auc_update_op_n, feed_dict=feed_dict_neg)
    sess.run(auc_update_op_p, feed_dict=feed_dict_pos)

    val_summaries_auc = sess.run(summaries_auc)

    train_writer.add_summary(val_summaries, step)
    train_writer.add_summary(val_summaries_auc, step)

##############
# Validation #
##############

logger.info("Starting validation")
sess.run(ini_l)
step = 0
for q, p, y in data_generator("test"):
    step += 1
    q_pos, p_pos = get_positives(q, p, y)
    q_neg, p_neg = get_negatives(q, p, y)

    feed_dict_pos = {
        query_raw: q_pos,
        product_raw: p_pos
    }

    feed_dict_neg = {
        query_raw: q_neg,
        product_raw: p_neg
    }

    sess.run(auc_update_op_n, feed_dict=feed_dict_neg)
    sess.run(auc_update_op_p, feed_dict=feed_dict_pos)

    val_summaries_auc = sess.run(summaries_auc)

    test_writer.add_summary(val_summaries_auc, step)

auc_pos_final = sess.run(auc_p)
auc_neg_final = sess.run(auc_n)

logger.info("Updating metrics...")
update_experiment(metarun_id=args.metarun_id, run_id=args.run_id,
                  metrics={
                      "auc_on_positives": np.float64(auc_pos_final),
                      "auc_on_negatives": np.float64(auc_neg_final)
                  })
logger.info("Updating done")
