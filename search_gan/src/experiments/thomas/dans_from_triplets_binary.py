from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import tensorflow as tf
import os
import logging
import functools
import time

# Setting path
root_dir = "{}/../..".format(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from common.flip_gradient import flip_gradient  # noqa
from common.experiment_db import update_experiment  # noqa
from common.io_utils import CircularRecordIterator, \
    batch_size, \
    nb_triplets_query_product_buckets  # noqa

start_time = time.time()

access_mode = "r"

# Parameters from grid
parser = argparse.ArgumentParser(
    description='Learn one classifier on each language, starting from a sparse representation of products')

parser.add_argument('--mode', type=str, default="source")

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
parser.add_argument('--dssm_units', type=str, default="50_50")
parser.add_argument('--discriminator_units', type=str, default="50")
parser.add_argument('--activate_dan_flow', type=int, default=0)
parser.add_argument('--source_id_and_target_id', type=str)
parser.add_argument('--max_step_for_training', type=float)
parser.add_argument('--max_step_for_validation', type=float)
parser.add_argument('--independent_query_embedding', type=int)
parser.add_argument('--gpu_id', type=int)

args, _ = parser.parse_known_args()

# Training - validation files location
cwd = os.getcwd()
log_root = f"{cwd}/logs/{args.experiment_name}"
try:
    os.mkdir(log_root)
except FileExistsError as e:
    print(f"Directory {log_root} already exists")
log_location = f"{log_root}/{args.metarun_id}_{args.run_id}_{args.mode}"

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


####################
# Model definition #
####################

class Model(object):

    @staticmethod
    def feature_extractor(inputs, item_type, p_keep_for_dropout):
        """
        Embedding
        """

        if args.independent_query_embedding == 0:
            scope_name = "feature_extractor"
        else:
            scope_name = f"feature_extractor_{item_type}"

        net = tf.layers.dense(
            inputs=inputs,
            units=args.embedding_size,
            activation=None,
            name=scope_name,
            reuse=tf.AUTO_REUSE)

        net = tf.layers.dropout(
            inputs=net,
            name=f"{scope_name}_dropout",
            rate=1 - p_keep_for_dropout)

        return net

    @staticmethod
    def dssm_atom(inputs, item_type, p_keep_for_dropout):
        """
        Helper function to build a two layer network for DSSM

        :param p_keep_for_dropout: p keep for dropout
        :param item_type: "product" or "query"
        :param inputs: input tensor
        """

        requested_units = [int(x) for x in args.dssm_units.split("_")]

        net = inputs

        for i, units in enumerate(requested_units):

            if i < len(requested_units) - 1:
                # Using relu for internal layers
                activation = tf.nn.relu
            else:
                # No activation for last layer
                activation = None

            net = tf.layers.dense(
                inputs=net,
                units=units,
                activation=activation,
                name=f"dssm_layer_{i}_{item_type}",
                reuse=tf.AUTO_REUSE
            )

            net = tf.layers.dropout(
                inputs=net,
                rate=1 - p_keep_for_dropout,
                name=f"dssm_layer_{i}_{item_type}_dropout"
            )

        return net

    @staticmethod
    def discriminator(inputs, p_keep_for_dropout):
        """
        Helper function to build a 2-layer discriminator

        :param p_keep_for_dropout:
        :param inputs: input tensor
        :return:
        """

        requested_units = [int(x) for x in args.discriminator_units.split("_")]

        net = inputs

        for i, units in enumerate(requested_units):
            net = tf.layers.dense(
                inputs=net,
                units=units,
                activation=tf.nn.relu,
                name=f"discriminator_layer_{i}",
                reuse=tf.AUTO_REUSE
            )

            net = tf.layers.dropout(
                inputs=net,
                rate=1 - p_keep_for_dropout,
                name=f"discriminator_layer_{i}_dropout"
            )

        net = tf.layers.dense(
            inputs=net,
            units=1,
            activation=tf.nn.sigmoid,
            name="discriminator_layer_2",
            reuse=tf.AUTO_REUSE
        )

        return net

    @staticmethod
    def reconstructor(inputs):

        requested_units = [int(x) for x in args.reconstructor_units.split("_")]

        net = inputs

        for i, units in enumerate(requested_units):
            net = tf.layers.dense(
                inputs=net,
                units=units,
                activation=tf.nn.relu,
                name=f"reconstructor_layer_{i}"
            )

        net = tf.layers.dense(
            inputs=net,
            units=nb_triplets_query_product_buckets,
            name="reconstructor_output_layer",
        )

        return net

    def __init__(self):
        self._build_graph()
        self._build_metrics()

    def _build_metrics(self):
        # Summaries
        with tf.name_scope("metrics"):
            self.dssm_auc, self.dssm_auc_update_op = tf.metrics.auc(
                labels=self.y_task,
                predictions=self.y_prediction_dssm_sigmoid,
                name="auc")

            self.summary_auc_valid = {
                "source": tf.summary.scalar("auc_valid_on_source", self.dssm_auc),
                "target": tf.summary.scalar("auc_valid_on_target", self.dssm_auc)
            }

            # Computing accuracy
            is_correct_discriminator = tf.equal(tf.sign(self.y_gans - 0.5),
                                                tf.sign(self.discriminator_prediction - 0.5))
            accuracy_discriminator = tf.reduce_mean(
                tf.cast(is_correct_discriminator, tf.float32))

        with tf.name_scope("summaries"):
            self.summary_dssm = tf.summary.merge([
                tf.summary.scalar("task_loss", self.task_loss),
                tf.summary.histogram("task_prediction", self.y_prediction_dssm_sigmoid),
                tf.summary.histogram("task_labels", self.y_task),
                tf.summary.histogram("QueryFastTextEmbeddings", self.query_source),
                tf.summary.histogram("ProductFastTextEmbeddings", self.product_source)
            ])

            self.summary_discriminator = tf.summary.merge([
                tf.summary.scalar("LogLossDiscriminator", self.discriminator_loss),
                tf.summary.scalar("AccuracyDiscriminator", accuracy_discriminator)
            ])

    def _build_graph(self):
        """
        Build the graph
        """
        with tf.device('/gpu:0'):
            ###################################
            # Placeholders for general inputs #
            ###################################

            with tf.name_scope("inputs"):
                # Dropout rate
                self.p_keep_for_dropout = tf.placeholder(tf.float32, shape=(),
                                                         name="dropout_rate")

                # Placeholders for product and query (triplets)
                self.product_source = tf.placeholder(tf.float32,
                                                     [None, nb_triplets_query_product_buckets],
                                                     name="product_source")
                self.query_source = tf.placeholder(tf.float32,
                                                   [None, nb_triplets_query_product_buckets],
                                                   name="query_source")

                self.product_target = tf.placeholder(tf.float32,
                                                     [None, nb_triplets_query_product_buckets],
                                                     name="product_target")

                # IsClicked label
                self.y_task = tf.placeholder(tf.float32, shape=(None, 1))

                # IsTarget label
                self.y_gans = tf.placeholder(tf.float32, shape=(None, 1))

            # Reshaping product and query

            ##############################
            # DSSM for product and query #
            ##############################

            with tf.name_scope("feature_extractor_product"):
                self.embedded_product_source = Model.feature_extractor(
                    inputs=self.product_source,
                    item_type="product",
                    p_keep_for_dropout=self.p_keep_for_dropout
                )

                self.embedded_product_target = Model.feature_extractor(
                    inputs=self.product_target,
                    item_type="product",
                    p_keep_for_dropout=self.p_keep_for_dropout
                )

            with tf.name_scope("feature_extractor_query"):
                self.embedded_query_source = Model.feature_extractor(
                    inputs=self.query_source,
                    item_type="query",
                    p_keep_for_dropout=self.p_keep_for_dropout
                )

            with tf.name_scope("dssm"):
                self.query_after_dssm_atom = Model.dssm_atom(
                    inputs=self.embedded_query_source,
                    item_type="query",
                    p_keep_for_dropout=self.p_keep_for_dropout
                )

                self.product_after_dssm_atom = Model.dssm_atom(
                    inputs=self.embedded_product_source,
                    item_type="product",
                    p_keep_for_dropout=self.p_keep_for_dropout
                )

                self.y_prediction_dssm = tf.reduce_sum(tf.multiply(
                    self.product_after_dssm_atom,
                    self.query_after_dssm_atom), 1, keepdims=True)

                # self.y_prediction_dssm = tf.keras.backend.batch_dot(
                #    self.product_after_dssm_atom,
                #    self.query_after_dssm_atom,
                #    axes=1
                # )

            self.dssm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    "dssm")
            logger.info("Variables for DSSM are {}".format(self.dssm_variables))

            self.feature_extractor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                 "feature_extractor")
            logger.info("Variables for feature extractor are {}".format(self.feature_extractor_variables))

            with tf.name_scope("metrics_dssm"):
                self.y_prediction_dssm_sigmoid = tf.nn.sigmoid(self.y_prediction_dssm)

            with tf.name_scope("loss_dssm"):
                self.task_loss = tf.losses.log_loss(
                    labels=self.y_task,
                    predictions=self.y_prediction_dssm_sigmoid
                )

                self.task_optimizer = tf.train.AdamOptimizer(
                    learning_rate=args.learning_rate)

                self.task_step = self.task_optimizer.minimize(
                    loss=self.task_loss,
                    var_list=self.dssm_variables + self.feature_extractor_variables)

            #################
            # Discriminator #
            #################

            with tf.name_scope("discriminator"):
                self.products_source_and_target = tf.concat(
                    values=[
                        self.embedded_product_source,
                        self.embedded_product_target
                    ],
                    axis=0
                )

                self.discriminator_prediction = Model.discriminator(
                    inputs=flip_gradient(self.products_source_and_target),
                    p_keep_for_dropout=self.p_keep_for_dropout
                )

            with tf.name_scope("loss_discriminator"):
                self.discriminator_loss = tf.losses.log_loss(
                    predictions=self.discriminator_prediction,
                    labels=self.y_gans
                )

                self.discriminator_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

                if args.activate_dan_flow == 1:
                    self.discriminator_variables += self.feature_extractor_variables
                logger.info(
                    "Variables for discriminator are {}".format(self.discriminator_variables))

                self.discriminator_optimizer = tf.train.AdamOptimizer(
                    learning_rate=args.learning_rate_dans
                )

                self.discriminator_step = self.discriminator_optimizer.minimize(
                    loss=self.discriminator_loss,
                    var_list=self.discriminator_variables)


############################################
# Session configuration and initialization #
############################################

model = Model()
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

if args.max_nb_processes > 1:
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.9 / args.max_nb_processes)
else:
    gpu_options = tf.GPUOptions()

config = tf.ConfigProto(
    allow_soft_placement=False,
    gpu_options=gpu_options,
    inter_op_parallelism_threads=8,
    intra_op_parallelism_threads=8
)

with tf.Session(config=config) as session:
    print(
        "Using log location for tensorboard {} (run_id is {}, metarun_id is {})".format(
            log_location, args.run_id,
            args.metarun_id))


    def get_writer(mode):
        return tf.summary.FileWriter(
            f"{log_location}_{mode}",
            session.graph)


    summary_writers_train = get_writer("train")
    summary_writers_test = get_writer("test")

    session.run(init)
    session.run(init_local)

    test_iterator = CircularRecordIterator(target_id, "test")


    def validate_on_test_target_batch():
        my_batch, _ = test_iterator.get_next_batch()

        feed_dict = {
            model.y_task: my_batch["labels"],
            model.query_source: my_batch["queries"],
            model.product_source: my_batch["products"],
            model.p_keep_for_dropout: 1.0,
        }

        val_summary_dssm = session.run(
            model.summary_dssm,
            feed_dict=feed_dict
        )

        summary_writers_test.add_summary(val_summary_dssm, step)


    if args.mode == "source":

        ######################
        # Learning on Source #
        ######################
        step = 0

        learning_iterator = CircularRecordIterator(source_id, "train")
        while True:

            my_batch, _ = learning_iterator.get_next_batch()

            feed_dict = {
                model.y_task: my_batch["labels"],
                model.query_source: my_batch["queries"],
                model.product_source: my_batch["products"],
                model.p_keep_for_dropout: args.p_keep_for_dropout
            }

            session.run(model.task_step, feed_dict=feed_dict)

            if step % 100 == 0:
                val_summary_dssm = session.run(model.summary_dssm, feed_dict=feed_dict)
                summary_writers_train.add_summary(val_summary_dssm, step)
                validate_on_test_target_batch()

            step += 1

            if step >= args.max_step_for_training:
                print(f"Max step reached for mode {args.mode}.")
                break

    elif args.mode == "target":

        ######################
        # Learning on Target #
        ######################

        step = 0
        learning_iterator = CircularRecordIterator(target_id, "train")
        while True:
            my_batch, _ = learning_iterator.get_next_batch()

            feed_dict = {
                model.y_task: my_batch["labels"],
                model.query_source: my_batch["queries"],
                model.product_source: my_batch["products"],
                model.p_keep_for_dropout: args.p_keep_for_dropout
            }

            session.run(model.task_step, feed_dict=feed_dict)

            if step % 100 == 0:
                val_summary_dssm = session.run(model.summary_dssm, feed_dict=feed_dict)
                summary_writers_train.add_summary(val_summary_dssm, step)
                validate_on_test_target_batch()

            step += 1

            if step >= args.max_step_for_training:
                print(f"Max step reached for mode {args.mode}.")
                break

    elif args.mode == "dann_product":

        ##############################
        # Using DANN on product only #
        ##############################

        step = 0
        source_iterator = CircularRecordIterator(source_id, "train")
        target_iterator = CircularRecordIterator(target_id, "train")

        while True:
            try:
                my_batch_source, _ = source_iterator.get_next_batch()
                my_batch_target, _ = target_iterator.get_next_batch()

                # Training for the task

                feed_dict = {
                    model.y_task: my_batch_source["labels"],
                    model.query_source: my_batch_source["queries"],
                    model.product_source: my_batch_source["products"],
                    model.p_keep_for_dropout: args.p_keep_for_dropout
                }

                session.run(model.task_step, feed_dict=feed_dict)

                if step % 100 == 0:
                    val_summary_dssm = session.run(model.summary_dssm, feed_dict=feed_dict)
                    summary_writers_train.add_summary(val_summary_dssm, step)
                    validate_on_test_target_batch()

                # Training for the DANN
                products_source_and_target = \
                    np.concatenate([
                        my_batch_source["products"],
                        my_batch_target["products"]
                    ], axis=0)

                label_domain = \
                    np.concatenate([
                        np.zeros((batch_size, 1)),
                        np.ones((batch_size, 1))
                    ], axis=0)

                feed_dict_dans = {
                    model.y_gans: label_domain,
                    model.product_source: my_batch_source["products"],
                    model.product_target: my_batch_target["products"],
                    model.p_keep_for_dropout: 1.0
                }

                session.run(model.discriminator_step, feed_dict=feed_dict_dans)

                if step % 100 == 0:
                    val_summary_discriminator = \
                        session.run(model.summary_discriminator, feed_dict=feed_dict_dans)
                    summary_writers_train.add_summary(val_summary_discriminator, step)

                step += 1

                if step >= args.max_step_for_training:
                    print(f"Max step reached for mode {args.mode}.")
                    break

            except tf.errors.OutOfRangeError:
                print(f"End of training dataset for mode {args.mode}.")
                break

    elif args.mode == "dann_reconstructor_pretrain":

        ################################################################
        # Using DANN on (product, query) with pretrained reconstructor #
        ################################################################

        raise NotImplementedError(f"{args.mode} not yet available !")

    elif args.mode == "dann_reconstructor_cotrain":

        ################################################################
        # Using DANN on (product, query) with cotrained reconstructor #
        ################################################################

        raise NotImplementedError(f"{args.mode} not yet available !")

    elif args.mode == "dann_reconstructor_partial_cotrain":

        #########################################################################
        # Using DANN on (product, query) with partially cotrained reconstructor #
        #########################################################################

        raise NotImplementedError(f"{args.mode} not yet available !")

    #############################
    # Validation of final model #
    #############################

    logger.info("Validating the learnt model on all domains")
    final_metrics = {
        "source": dict(),
        "target": dict()
    }

    summary_writers_valid = get_writer("valid")

    validation_iterators = {
        "source": CircularRecordIterator(source_id, "val"),
        "target": CircularRecordIterator(target_id, "val")
    }

    for validation_domain in ["source", "target"]:
        step = 0

        # Clean local variable for metrics
        session.run(init_local)

        validation_iterator = validation_iterators[validation_domain]

        while True:

            try:
                my_batch, _ = validation_iterator.get_next_batch()

                feed_dict = {
                    model.y_task: my_batch["labels"],
                    model.query_source: my_batch["queries"],
                    model.product_source: my_batch["products"],
                    model.p_keep_for_dropout: 1.0
                }

                session.run(
                    model.dssm_auc_update_op,
                    feed_dict=feed_dict
                )

                if step % 100 == 0:
                    sum_value = session.run(model.summary_auc_valid[validation_domain])
                    summary_writers_valid.add_summary(sum_value, step)

                step += 1

                if step >= args.max_step_for_validation:
                    print(f"Max step reached for validation on {validation_domain}.")
                    break

            except tf.errors.OutOfRangeError:
                print(f"End of validation dataset for domain {validation_domain}.")
                break

        final_metrics[validation_domain] = {
            "auc": np.float64(session.run(model.dssm_auc))
        }
    final_metrics["elapsed_time"] = time.time() - start_time
    final_metrics["status"] = "completed"

    logger.info(final_metrics)

    logger.info("Updating metrics...")
    update_experiment(metarun_id=args.metarun_id, run_id=args.run_id,
                      metrics=final_metrics)
    logger.info("Updating done")
