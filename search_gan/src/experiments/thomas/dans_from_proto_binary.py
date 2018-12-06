from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import tensorflow as tf
import os
import logging
import time

# Setting path
root_dir = "{}/../..".format(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from common.flip_gradient import flip_gradient  # noqa
from common.experiment_db import update_experiment  # noqa
from common.io_utils import CircularRecordIterator, parse_serialized_batch, nb_triplets_query_product_buckets  # noqa

start_time = time.time()

access_mode = "r"

# Training - validation files location
cwd = os.getcwd()
log_location = cwd + '/logs/'

# Parameters from grid
parser = argparse.ArgumentParser(
    description='Learn one classifier on each language, starting from a sparse representation of products')

parser.add_argument('--learning_rate', type=float, default=0.00005,
                    metavar='N')
parser.add_argument('--learning_rate_dans', type=float, default=0.00005,
                    metavar='N')
parser.add_argument('--p_keep_for_dropout', type=float)
parser.add_argument('--run_id', type=int, default=0, metavar='N')
parser.add_argument('--metarun_id', type=int, default=0, metavar='N')
parser.add_argument('--max_nb_processes', type=int)
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--noise_level_for_domain_labels', type=float)
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--dssm_layer_1_size', type=int, default=300)
parser.add_argument('--dssm_layer_2_size', type=int, default=100)
parser.add_argument('--discriminator_layer_size', type=int, default=50)
parser.add_argument('--activate_dan_flow', type=int, default=0)
parser.add_argument('--source_id_and_target_id', type=str)
parser.add_argument('--max_epoch_for_training', type=float)
parser.add_argument('--independent_query_embedding', type=int)
parser.add_argument('--gpu_id', type=int)

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

source_id, target_id = args.source_id_and_target_id.split("_")

logger.info(
    "Starting job (source={}, target={})...".format(source_id, target_id))


def embedding(input_x, model_type, typ, p_keep_for_dropout):
    """
    Embedding
    """

    if args.independent_query_embedding == 0:
        scope_name = "embedding_{}".format(model_type)
    else:
        scope_name = "embedding_{}_{}".format(model_type, typ)

    # Number of samples in the input
    n_input = input_x.get_shape().as_list()[1]

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(
            name='W',
            shape=[n_input, args.embedding_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[args.embedding_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

    return tf.nn.dropout(tf.matmul(input_x, w) + b,
                         keep_prob=p_keep_for_dropout)


def dssm_atom(input_x, typ, domain, p_keep_for_dropout):
    """
    Helper function to build a two layer network for DSSM

    :param p_keep_for_dropout: p keep for dropout
    :param domain: "source" or "target"
    :param typ: "product" or "query"
    :param input_x: input tensor
    """

    # Number of samples in the input
    n_input = input_x.get_shape().as_list()[1]

    with tf.variable_scope("dssm_{d}_{t}".format(d=domain, t=typ),
                           reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable(
            name='W1',
            shape=[n_input, args.dssm_layer_1_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable(
            name='b1',
            shape=[args.dssm_layer_1_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        w2 = tf.get_variable(
            name='W2',
            shape=[args.dssm_layer_1_size, args.dssm_layer_2_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b2 = tf.get_variable(
            name='b2',
            shape=[args.dssm_layer_2_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

    x1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
    x1_dropout = tf.nn.dropout(x1, keep_prob=p_keep_for_dropout)
    return tf.nn.dropout(tf.matmul(x1_dropout, w2) + b2,
                         keep_prob=p_keep_for_dropout)


def discriminator(input_x):
    """
    Helper function to build a 2-layer discriminator

    :param input_x: input tensor
    :return:
    """
    # Number of samples in the input
    n_input = input_x.get_shape().as_list()[1]

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable(
            name='W1',
            shape=[n_input, args.discriminator_layer_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable(
            name='b1',
            shape=[args.discriminator_layer_size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        w2 = tf.get_variable(
            name='W2',
            shape=[args.discriminator_layer_size, 1],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b2 = tf.get_variable(
            name='b2',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

    x1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
    return tf.nn.sigmoid(tf.matmul(x1, w2) + b2)


def flatten_and_reshape(input_x):
    x_flat = tf.reshape(input_x,
                        shape=[-1, nb_triplets_query_product_buckets],
                        name="{}_flat".format(input_x.name.split(":")[0]))
    return tf.concat([x_flat], 1)


####################
# Graph definition #
####################


with tf.device('/gpu:0'):
    ###################################
    # Placeholders for general inputs #
    ###################################

    # Model type (source / source_dans / target)
    model_type = tf.placeholder(tf.string, shape=(), name="model_type")

    # Dropout rate
    p_keep_for_dropout = tf.placeholder(tf.float32, shape=(),
                                        name="dropout_rate")

    # Placeholders for product and query (triplets)
    product_input = tf.placeholder(tf.float32,
                                   [None, nb_triplets_query_product_buckets,
                                    1],
                                   name="productname_triplets_dssm")
    query_input = tf.placeholder(tf.float32,
                                 [None, nb_triplets_query_product_buckets, 1],
                                 name="query_triplets")

    # IsClicked label
    y_task = tf.placeholder(tf.float32, shape=(None, 1))

    # IsTarget label
    y_gans = tf.placeholder(tf.float32, shape=(None, 1))

    # Reshaping product and query
    product = flatten_and_reshape(product_input)
    query = flatten_and_reshape(query_input)

    ##############################
    # DSSM for product and query #
    ##############################

    embedded_product_source = embedding(product, model_type="source",
                                        typ="product",
                                        p_keep_for_dropout=p_keep_for_dropout)
    embedded_product_dans = embedding(product, model_type="source_dans",
                                      typ="product",
                                      p_keep_for_dropout=p_keep_for_dropout)
    embedded_product_target = embedding(product, model_type="target",
                                        typ="product",
                                        p_keep_for_dropout=p_keep_for_dropout)

    embedded_query_source = embedding(query, model_type="source", typ="query",
                                      p_keep_for_dropout=p_keep_for_dropout)
    embedded_query_dans = embedding(query, model_type="source_dans",
                                    typ="query",
                                    p_keep_for_dropout=p_keep_for_dropout)
    embedded_query_target = embedding(query, model_type="target", typ="query",
                                      p_keep_for_dropout=p_keep_for_dropout)

    query_after_dssm_atom = tf.cond(
        pred=tf.equal(model_type, tf.constant('source')),
        true_fn=lambda: dssm_atom(embedded_query_source, typ="query",
                                  domain="source",
                                  p_keep_for_dropout=p_keep_for_dropout),
        false_fn=lambda: tf.cond(
            pred=tf.equal(model_type, tf.constant('source_dans')),
            true_fn=lambda: dssm_atom(embedded_query_dans,
                                      typ="query", domain="source_dans",
                                      p_keep_for_dropout=p_keep_for_dropout),
            false_fn=lambda: dssm_atom(embedded_query_target, typ="query",
                                       domain="target",
                                       p_keep_for_dropout=p_keep_for_dropout)))

    product_after_dssm_atom = tf.cond(
        pred=tf.equal(model_type, tf.constant('source')),
        true_fn=lambda: dssm_atom(embedded_product_source, typ="product",
                                  domain="source",
                                  p_keep_for_dropout=p_keep_for_dropout),
        false_fn=lambda: tf.cond(
            pred=tf.equal(model_type, tf.constant('source_dans')),
            true_fn=lambda: dssm_atom(embedded_product_dans, typ="product",
                                      domain="source_dans",
                                      p_keep_for_dropout=p_keep_for_dropout),
            false_fn=lambda: dssm_atom(embedded_product_target,
                                       typ="product",
                                       domain="target",
                                       p_keep_for_dropout=p_keep_for_dropout)))

    # dssm prediction
    y_prediction_dssm = tf.reduce_sum(tf.multiply(
        query_after_dssm_atom,
        product_after_dssm_atom), 1, keepdims=True)

    # Loss for dssm
    # cross_entropy_dssm = tf.reduce_mean(
    #    tf.losses.log_loss(labels=y_task, predictions=y_prediction_dssm), name="XEntropy_dssm")

    cross_entropy_dssm = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_task,
                                                logits=y_prediction_dssm),
        name="cross_entropy_dssm")

    dssm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "dssm")
    logger.info("Variables for DSSM are {}".format(dssm_variables))

    embedding_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            "embedding")
    logger.info("Variables for embedding are {}".format(embedding_variables))

    # AUCs

    y_prediction_dssm_sigm = tf.nn.sigmoid(y_prediction_dssm)

    dssm_aucs = {
        "source_dans": tf.metrics.auc(labels=y_task,
                                      predictions=y_prediction_dssm_sigm,
                                      name="auc_dans"),
        "source": tf.metrics.auc(labels=y_task,
                                 predictions=y_prediction_dssm_sigm,
                                 name="auc_source"),
        "target": tf.metrics.auc(labels=y_task,
                                 predictions=y_prediction_dssm_sigm,
                                 name="auc_target")
    }

    # dssm Loss minimization step
    adam_train_step_dssm = tf.train.AdamOptimizer(
        learning_rate=args.learning_rate).minimize(cross_entropy_dssm,
                                                   var_list=dssm_variables + embedding_variables)
    summary_dssm = tf.summary.merge([
        tf.summary.scalar("XEntropy_DSSM", cross_entropy_dssm),
        tf.summary.histogram("Prediction_DSSM", y_prediction_dssm_sigm),
        tf.summary.histogram("InputLabels_DSSM", y_task)
    ])

    #################
    # Discriminator #
    #################

    y_gans_pred = discriminator(flip_gradient(embedded_product_dans))

    discriminator_loss = tf.reduce_sum(
        tf.losses.log_loss(predictions=y_gans_pred, labels=y_gans))

    discriminator_variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    if args.activate_dan_flow == 1:
        discriminator_variables += embedding_variables
    logger.info(
        "Variables for discriminator are {}".format(discriminator_variables))

    adam_train_step_discriminator = tf.train.AdamOptimizer(
        learning_rate=args.learning_rate_dans).minimize(
        discriminator_loss, var_list=discriminator_variables)

    # Computing accuracy
    is_correct_discriminator = tf.equal(tf.sign(y_gans - 0.5),
                                        tf.sign(y_gans_pred - 0.5))
    accuracy_discriminator = tf.reduce_mean(
        tf.cast(is_correct_discriminator, tf.float32))

    summary_discriminator = tf.summary.merge([
        tf.summary.scalar("LogLossDiscriminator", discriminator_loss),
        tf.summary.scalar("AccuracyDiscriminator", accuracy_discriminator)
    ])

############################################
# Session configuration and initialization #
############################################

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

if args.max_nb_processes > 1:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.9 / args.max_nb_processes)
else:
    gpu_options = tf.GPUOptions()

config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=config) as session:
    session.run(init)
    session.run(init_local)
    log_location = log_location + args.experiment_name
    print(
        "Using log location for tensorboard {} (run_id is {}, metarun_id is {})".format(
            log_location, args.run_id,
            args.metarun_id))
    try:
        os.mkdir(log_location)
    except FileExistsError as e:
        print("Directory already exists")


    def get_writer(mode):
        return tf.summary.FileWriter(
            log_location + '/{}_{}_{}'.format(args.metarun_id, args.run_id,
                                              mode),
            session.graph)


    summary_writers_train = {
        "source": get_writer("source_train"),
        "target": get_writer("target_train"),
        "source_dans": get_writer("source_dans_train")
    }

    summary_writers_test = {
        "source": get_writer("source_test"),
        "target": get_writer("target_test"),
        "source_dans": get_writer("source_dans_test")
    }

    iterators = {
        "train": {
            "source": CircularRecordIterator(source_id, "train"),
            "target": CircularRecordIterator(target_id, "train"),
            "dans_source": CircularRecordIterator(source_id, "train"),
            "dans_target": CircularRecordIterator(target_id, "train")
        },
        "test": {
            "source": CircularRecordIterator(source_id, "test"),
            "target": CircularRecordIterator(target_id, "test")
        },
        "validation": {
            "source": CircularRecordIterator(source_id, "val"),
            "target": CircularRecordIterator(target_id, "val")
        }
    }

    step = 0
    epochs = {
        "source": 1,
        "target": 1,
        "dans_source": 1,
        "dans_target": 1
    }

    keep_iterating = True

    while keep_iterating:
        step += 1

        ##############################
        # Training source and target #
        ##############################

        for model_typ in ["source", "target"]:

            # if epochs[model_typ] <= args.max_epoch_for_training:
            serialized_data, is_new_epoch = iterators["train"][
                model_typ].get_next_serialized_batch()
            q, p, y = parse_serialized_batch(serialized_data=serialized_data)

            if is_new_epoch:
                logger.info(
                    "Finished epoch {} for {}".format(epochs[model_typ],
                                                      model_typ))
                epochs[model_typ] = epochs[model_typ] + 1

            feed_dict = {
                y_task: y,
                query_input: q,
                product_input: p,
                p_keep_for_dropout: args.p_keep_for_dropout,
                model_type: model_typ
            }

            val_summary_dssm, _ = session.run(
                [summary_dssm, adam_train_step_dssm],
                feed_dict=feed_dict)

            if step % 100 == 0:
                summary_writers_train[model_typ].add_summary(val_summary_dssm,
                                                             step)

        ########################
        # Training source_dans #
        ########################

        if min(epochs["dans_target"],
               epochs["dans_source"]) <= args.max_epoch_for_training:
            model_typ = "source_dans"
            serialized_data, is_new_epoch_source = iterators["train"][
                "dans_source"].get_next_serialized_batch()
            q_source, p_source, y_source = parse_serialized_batch(
                serialized_data=serialized_data)

            serialized_data, is_new_epoch_target = iterators["train"][
                "dans_target"].get_next_serialized_batch()
            _, p_target, _ = parse_serialized_batch(
                serialized_data=serialized_data)

            if is_new_epoch_source:
                logger.info("Finished epoch {} for dans_source".format(
                    epochs["dans_source"]))
                epochs["dans_source"] = epochs["dans_source"] + 1

            if is_new_epoch_target:
                logger.info("Finished epoch {} for dans_target".format(
                    epochs["dans_target"]))
                epochs["dans_target"] = epochs["dans_target"] + 1

            feed_dict = {
                y_task: y_source,
                query_input: q_source,
                product_input: p_source,
                p_keep_for_dropout: args.p_keep_for_dropout,
                model_type: model_typ
            }

            val_summary_dssm, _ = session.run(
                [summary_dssm, adam_train_step_dssm],
                feed_dict=feed_dict)

            if step % 100 == 0:
                summary_writers_train[model_typ].add_summary(val_summary_dssm,
                                                             step)

            p_target_and_source = np.concatenate([p_source, p_target], axis=0)
            labels_for_dans = np.concatenate(
                [np.zeros((32, 1)), np.ones((32, 1))], axis=0)

            feed_dict_dans = {
                y_gans: labels_for_dans,
                product_input: p_target_and_source,
                p_keep_for_dropout: 1.0
            }

            val_y_real, val_y_pred = session.run([y_gans, y_gans_pred],
                                                 feed_dict=feed_dict_dans)

            val_summary_discr, _ = session.run(
                [summary_discriminator, adam_train_step_discriminator],
                feed_dict=feed_dict_dans)

            if step % 100 == 0:
                summary_writers_train[model_typ].add_summary(val_summary_discr,
                                                             step)

        # If all the learning are done, we can stop iterating
        if min([
            min(epochs["dans_target"], epochs["dans_source"]),
            epochs["source"],
            epochs["target"]
        ]) > args.max_epoch_for_training:
            keep_iterating = False
            logger.info("All learning done.")

        ##############
        # Validation #
        ##############

        if step % 100 == 0:
            serialized_data, _ = iterators["test"][
                "target"].get_next_serialized_batch()
            q_target, p_target, y_target = parse_serialized_batch(
                serialized_data=serialized_data)

            for model_typ in ["source", "target", "source_dans"]:
                feed_dict = {
                    y_task: y_target,
                    query_input: q_target,
                    product_input: p_target,
                    p_keep_for_dropout: 1.0,
                    model_type: model_typ
                }

                val_summary_dssm = session.run(
                    [summary_dssm],
                    feed_dict=feed_dict)[0]

                summary_writers_test[model_typ].add_summary(val_summary_dssm,
                                                            step)

    #############################
    # Validation of final model #
    #############################

    logger.info("Validating all the learnt models on target and source domain")
    final_metrics = {
        model_typ: dict() for model_typ in ["source", "target", "source_dans"]
    }

    for validation_domain in ["source", "target"]:

        step = 0
        keep_iterating = True
        # Clean local variable
        session.run([init_local])
        while keep_iterating:

            serialized_data, is_new_epoch = iterators["validation"][
                validation_domain].get_next_serialized_batch()
            q_target, p_target, y_target = parse_serialized_batch(
                serialized_data=serialized_data)

            if is_new_epoch:
                keep_iterating = False
                break

            for model_typ in ["source", "target", "source_dans"]:
                feed_dict = {
                    y_task: y_target,
                    query_input: q_target,
                    product_input: p_target,
                    p_keep_for_dropout: 1.0,
                    model_type: model_typ
                }

                dssm_auc, dssm_auc_update_op = dssm_aucs[model_typ]

                xentropy, _ = session.run(
                    [cross_entropy_dssm, dssm_auc_update_op],
                    feed_dict=feed_dict)

                final_metrics[model_typ][
                    "xentropy_{}".format(validation_domain)] = \
                    final_metrics[model_typ].get(
                        "xentropy_{}".format(validation_domain), 0) + xentropy

                step += 1

        # Finalizing metrics
        for model_typ in ["source", "target", "source_dans"]:
            # Reading final auc
            dssm_auc = dssm_aucs[model_typ][0]
            final_auc = session.run([dssm_auc])[0]
            final_metrics[model_typ][
                "auc_{}".format(validation_domain)] = np.float64(final_auc)

            # Normalizing cross-entropy
            final_metrics[model_typ]["xentropy_{}".format(validation_domain)] = \
                np.float64(final_metrics[model_typ][
                               "xentropy_{}".format(validation_domain)] / step)

    final_metrics["elapsed_time"] = time.time() - start_time
    final_metrics["status"] = "completed"

    logger.info(final_metrics)

    logger.info("Updating metrics...")
    update_experiment(metarun_id=args.metarun_id, run_id=args.run_id,
                      metrics=final_metrics)
    logger.info("Updating done")
