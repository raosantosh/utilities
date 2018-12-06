from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import shutil

import mmh3
import time
from sets import Set

import numpy as np
import tensorflow as tf
import os
from myutils import CircularFile

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
access_mode = "r"

# Training - validation files location
cwd = os.getcwd()
log_location = cwd + '/../logs/'
path_data = '/var/opt/amin/Data/datasets/'
path_model = cwd + '/../resources/datasets/models/'
positives_training_file = path_data + "positive_training_samples_query_productname_stemmed_193.csv"
negatives_training_file = path_data + "negative_training_samples_query_productname_stemmed_193.csv"
test_positives_file = path_data + "test_positives_file_131"
test_negatives_file = path_data + "test_negatives_file_131"
positives_validation_file = path_data + "positive_training_samples_query_productname_stemmed_101.csv"
negatives_validation_file = path_data + "negative_training_samples_query_productname_stemmed_101.csv"
max_positive_training_samples_size = 11381259 - 381259
max_negative_training_samples_size = 27497519 - 497519

queries_source_embeddings = path_data + "queries_101_embeddings.csv"
queries_target_embeddings = path_data + "queries_193_embeddings.csv"

product_source_positive_embeddings = path_data + "positve_product_names_embeddings_101.csv"
product_target_positive_embeddings = path_data + "positve_product_names_embeddings_193.csv"

product_source_negative_embeddings = path_data + "product_names_101_negatives_embeddings.csv"
product_target_negative_embeddings = path_data + "product_names_193_negatives_embeddings.csv"



# p_fptr = CircularFile(positives_training_file)
# n_fptr = CircularFile(negatives_training_file)
# fp_val = CircularFile(positives_validation_file)
# fn_val = CircularFile(negatives_validation_file)


queries_source_embeddings_file =  CircularFile(queries_source_embeddings)
queries_target_embeddings_file =  CircularFile(queries_target_embeddings)
product_source_positive_embeddings_file =  CircularFile(product_source_positive_embeddings)
product_target_positive_embeddings_file =  CircularFile(product_target_positive_embeddings)
product_source_negative_embeddings_file =  CircularFile(product_source_negative_embeddings)
product_target_negative_embeddings_file =  CircularFile(product_target_negative_embeddings)

# Parameteres

embeddings_size = 100

learning_rate = 0.0001
droupout_rate = 0.4
first_layer_size = 10  # DSSM paper
second_layer_size = 3  # DSSM paper
second_layer_size_multi_domain = 1000
batch_positives = 8
negative_samples_factor = 32
query_max_length = 25
productname_max_length = 90
brand_max_length = 25
#nb_triplets_query_product_buckets = 2 ** 13
nb_triplets_query_product_buckets = embeddings_size
batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 100
QUERY_INDEX = 0
PRODCUTNAME_INDEX_POSITIVE = 1
PRODCUTNAME_INDEX_NEGATIVE = 0
PRODUCTDESC_INDEX_POSITIVE = 2
PRODUCTDESC_INDEX_NEGATIVE = 1
PRODUCTBRAND_INDEX_POSITIVE = 3
PRODUCTBRAND_INDEX_NEGATIVE = 2


def normalizeQuery(query, maxlength):
    query = query.replace("+", " ")
    query = query.replace("|", "")
    query = query.strip()
    query = query.lower()
    if len(query) > maxlength:
        query = query[0:maxlength]
    query = query.strip()
    return query;


def normalizeProduct(product, maxlength):
    product = product.replace("&apos;", "")
    product = product.replace("&nbsp;", "")
    product = product.replace("&ndash;", "")
    product = product.replace("&reg;", "")
    product = product.replace("&rsquo;", "")
    product = product.replace("&#38;", "")
    product = product.replace("&#39;", "")
    product = product.replace("&#40;", "")
    product = product.replace("&#41;", "")
    product = product.replace("&#45;", "")
    product = product.replace("&#46;", "")
    product = product.replace("&#47;", "")
    product = product.replace("&#143;", "")
    product = product.replace("&#153;", "")
    product = product.replace("&#160;", "")
    product = product.replace("&#169;", "")
    product = product.replace("&#174;", "")
    product = product.replace("&#176;", "")
    product = product.replace("&#180;", "")
    product = product.replace("&#232;", "")
    product = product.replace("&#233;", "")
    product = product.replace("&@174;", "")
    product = product.replace("|", "")
    if len(product) > maxlength:
        product = product[0:maxlength]

    product = product.lower()
    product = product.strip()
    return product


def getTriplets(query, length):
    triplets = Set()
    tokens = query.rstrip().split(' ')
    for token in tokens:
        token = "#" + token + "#"
        for i in range(len(token) - length + 1):
            triplets.add(token[i:i + length])
            # triplets.add(token[i:i + length+1])
    return triplets


def getinputtriplets(input, len, BUCKETS):
    features = Set()
    for triplet in getTriplets(input, len):
        features.add(abs(int(mmh3.hash(triplet))) % BUCKETS)
    return features


def query2producttripletrepresentation(query, product, BUCKETS):
    features = Set()
    qgrams_4 = getTriplets(query, 3)
    pgrams_4 = getTriplets(product, 3)
    for gram in qgrams_4:
        if gram in pgrams_4:
            features.add(abs(int(mmh3.hash(gram))) % BUCKETS)
    return features


def get_next_test_data_embeddings(positives_count, test_negative_samples_factor, queries_pos, product_pos, product_neg):
    test_data_size = positives_count * (test_negative_samples_factor + 1)
    test_labels = np.zeros((test_data_size, 1))
    query_triplets = np.zeros((test_data_size, embeddings_size, 1))
    productname_triplets = np.zeros((test_data_size, embeddings_size, 1))

    product_index = 0
    for index in range(positives_count):
        positives_line = queries_pos.readline()
        positives_line = positives_line.split(' ')
        for i in range(embeddings_size):
            query_triplets[product_index, i, 0] = float(positives_line[i])

        positives_line = product_pos.readline()
        positives_line = positives_line.split(' ')
        for i in range(embeddings_size):
            productname_triplets[product_index, i, 0] = float(positives_line[i])

        product_index += 1
        negatives = 0

        while (negatives != test_negative_samples_factor):
            negative_line = product_neg.readline()
            negative_line = negative_line.split(' ')

            for i in range(embeddings_size):
                query_triplets[product_index, i, 0] = float(positives_line[i])

            for i in range(embeddings_size):
                productname_triplets[product_index, i, 0] = float(negative_line[i])
            negatives+=1
            product_index+=1

    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index, 0] = 1


    return test_labels, query_triplets, productname_triplets


def get_next_test_data(positives_count, test_negative_samples_factor, fpp, fnn):
    # test_negative_samples_factor = 1  # negative_samples_factor
    test_data_size = positives_count * (test_negative_samples_factor + 1)
    test_labels = np.zeros((test_data_size, 1))

    query_productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    # query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    # query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    product_index = 0

    for index in range(positives_count):
        positives_line = fpp.readline()
        ptokens = positives_line.rstrip().split('|')
        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;
        querynormalized = normalizeQuery(ptokens[QUERY_INDEX], query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)

        features_query_triplets = getinputtriplets(querynormalized, 3, nb_triplets_query_product_buckets)
        features_productname_triplets = getinputtriplets(productnamenormalized, 3,
                                                         nb_triplets_query_product_buckets)
        features_query_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized,
                                                                                 nb_triplets_query_product_buckets)
        for cidx in features_query_triplets:
            query_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_triplets:
            productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_query_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        product_index += 1
        negatives = 0

        while (negatives != test_negative_samples_factor):
            negatives_line = fnn.readline()
            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;
            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            features_query_triplets = getinputtriplets(querynormalized, 3, nb_triplets_query_product_buckets)
            features_productname_triplets = getinputtriplets(productnamenormalized, 3,
                                                             nb_triplets_query_product_buckets)
            features_query_productname_triplets = query2producttripletrepresentation(querynormalized,
                                                                                     productnamenormalized,
                                                                                     nb_triplets_query_product_buckets)
            for cidx in features_query_triplets:
                query_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_triplets:
                productname_triplets[product_index, cidx, 0] = 1
            for cidx in features_query_productname_triplets:
                query_productname_triplets[product_index, cidx, 0] = 1
            product_index += 1
            negatives += 1
    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index, 0] = 1
    return test_labels, query_productname_triplets, query_triplets, productname_triplets


def phi(x, n_output, droupout_rate, isTraining, name=None, batch_normalization=None, activation=None, reuse=None,
        dropout=None):
    n_input = x.get_shape().as_list()[1]
    bn_epsilon = 1e-3
    normal_axes = [0]

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if batch_normalization:
            mu, sigma_squared = tf.nn.moments(h, normal_axes)
            h = (h - mu) / tf.sqrt(sigma_squared + bn_epsilon)
            gamma = tf.Variable(np.ones(n_output, np.float32), name='gamma')
            beta = tf.Variable(np.zeros(n_output, np.float32), name='beta')

            h = gamma * h + beta

        if activation:
            h = activation(h)

        if dropout:
            h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                        lambda: tf.layers.dropout(h, rate=0.0, training=True))

    return h, W


def multiphi(parameter_group, input, n_output, droupout_rate, isTraining, name=None, batch_normalization=None,
             activation=None, dropout=None,
             reuse=None):
    query_product_out_2, _ = \
        tf.cond(tf.equal(parameter_group, tf.constant('source')),
                lambda: phi(input, n_output=n_output, droupout_rate=droupout_rate,
                            activation=activation, name=name + 'source',
                            isTraining=isTraining,
                            dropout=dropout, reuse=reuse),
                lambda: tf.cond(tf.equal(parameter_group, tf.constant('target')),
                                lambda: phi(input, n_output=n_output, droupout_rate=droupout_rate,
                                            activation=activation, name=name + 'target',
                                            isTraining=isTraining,
                                            dropout=dropout, reuse=reuse),
                                lambda: tf.cond(tf.equal(parameter_group, tf.constant('domain_classifier_LR')),
                                                lambda: phi(input, n_output=n_output, droupout_rate=droupout_rate,
                                                            activation=activation, name=name + 'domain_classifier_LR',
                                                            isTraining=isTraining,
                                                            dropout=dropout, reuse=reuse),
                                                lambda: tf.cond(tf.equal(parameter_group, tf.constant('source_pr')),
                                                                lambda: phi(input, n_output=n_output,
                                                                            droupout_rate=droupout_rate,
                                                                            activation=activation,
                                                                            name=name + 'source_pr',
                                                                            batch_normalization=batch_normalization,
                                                                            isTraining=isTraining,
                                                                            dropout=dropout, reuse=reuse),
                                                                lambda: phi(input, n_output=n_output,
                                                                            droupout_rate=droupout_rate,
                                                                            activation=activation,
                                                                            name=name + 'discriminator',
                                                                            batch_normalization=batch_normalization,
                                                                            isTraining=isTraining,
                                                                            dropout=dropout, reuse=reuse)
                                                                )
                                                )
                                )
                )
    return query_product_out_2


def reuseNet(input_x):
    product_out_target_1, _ = phi(input_x, n_output=second_layer_size_multi_domain,
                                  droupout_rate=droupout_rate,
                                  activation=None, name='generator_l0',
                                  isTraining=isTrain,
                                  dropout=False,
                                  reuse=tf.AUTO_REUSE)
    product_out_target_2, _ = phi(product_out_target_1, n_output=nb_triplets_query_product_buckets,
                                  droupout_rate=droupout_rate,
                                  activation=None, name='generator_l1',
                                  isTraining=isTrain,
                                  dropout=False,
                                  reuse=tf.AUTO_REUSE)
    return product_out_target_2


with tf.device('/cpu:0'):
    # query-product classifier
    max_iterations = 20
    num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))
    isTrain = tf.placeholder(tf.bool, shape=(), name="isTrain")
    isTargetProduct = tf.placeholder(tf.bool, shape=(), name="isTargetProduct")
    name_training_domain = tf.placeholder(tf.string, shape=(), name="domain")

    # Letter nGram Model LR
    # query_productname_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
    #                                                 name="query_productname_triplets")
    # query_productname_triplets_emb_flat = tf.reshape(query_productname_triplets_emb,
    #                                                  [-1, nb_triplets_query_product_buckets],
    #                                                  name="query_productname_triplets_flat")
    #
    # query_product = tf.concat([query_productname_triplets_emb_flat], 1)
    y_true = tf.placeholder(tf.float32, shape=(None, 1))
    #
    # query_product_out_2 = multiphi(name_training_domain, query_product, n_output=1, droupout_rate=droupout_rate,
    #                                activation=None, name='query_product_out_2',
    #                                isTraining=isTrain,
    #                                dropout=True)
    #
    # y_prediction = query_product_out_2
    # pCTR = tf.nn.sigmoid(y_prediction, name="pCTR")
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction))
    # cross_entropy_summary = tf.summary.scalar("Cross entropy", cross_entropy)
    # accuracy_domain, accuracy_domain_op = tf.metrics.accuracy(y_true, tf.cast(tf.greater_equal(pCTR, 0.5), tf.float32))
    # accuracy_domain_summary = tf.summary.scalar("Accuracy domain classifier", accuracy_domain)
    # adam_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # projection of target product
    # productname_target_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
    #                                          name="productname_target_triplets_dssm")
    # # only negative products are projected
    # productname_target_triplets_emb_negatives = tf.gather(productname_target_triplets_emb,tf.gather(tf.where(tf.equal(y_true,0)),0,axis=1))
    # productname_target_triplets_emb_flat = tf.reshape(productname_target_triplets_emb, [-1, nb_triplets_query_product_buckets],
    #                                            name="productname_target_triplets_dssm_flat")

    # product_target = tf.concat([productname_target_triplets_emb_flat], 1)

    # dssm

    query_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                        name="query_triplets")
    query_triplets_emb_flat = tf.reshape(query_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                         name="query_triplets_flat")

    query_out_1 = multiphi(name_training_domain, query_triplets_emb_flat, n_output=first_layer_size,
                           droupout_rate=droupout_rate,
                           activation=tf.nn.tanh, name='qp_triplets_emb_projection_q1',
                           isTraining=isTrain,
                           dropout=True)
    query_out_2 = multiphi(name_training_domain, query_out_1, n_output=second_layer_size,
                           droupout_rate=droupout_rate,
                           activation=tf.nn.tanh, name='qp_triplets_emb_projection_q2',
                           isTraining=isTrain,
                           dropout=True)

    productname_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="productname_triplets_dssm")
    productname_triplets_emb_flat = tf.reshape(productname_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                               name="productname_triplets_dssm_flat")

    product = tf.concat([productname_triplets_emb_flat], 1)

    product_out_1 = tf.cond(isTrain,
                            lambda: product,
                            lambda: tf.cond(isTargetProduct,
                                            lambda: reuseNet(product),
                                            lambda: product
                                            )
                            )

    product_out_2 = multiphi(name_training_domain, product_out_1, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='qp_triplets_emb_projection_p1',
                             isTraining=isTrain,
                             dropout=True)

    product_out_3 = multiphi(name_training_domain, product_out_2, n_output=second_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='qp_triplets_emb_projection_p2',
                             isTraining=isTrain,
                             dropout=True)
    # y_true = tf.placeholder(tf.float32, shape=(None, 1))
    y_prediction_dssm = tf.reduce_sum(tf.multiply(query, product), 1, keep_dims=True)
    cross_entropy_dssm = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction_dssm), name="cross_entropy_dssm")
    cross_entropy_dssm_summary = tf.summary.scalar("Cross entropy DSSM", cross_entropy_dssm)
    variables_projections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              "qp_triplets_emb_projection")
    adam_train_step_dssm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_dssm,
                                                                                        var_list=variables_projections)

    # tf.cond(isTargetProduct,
    #                            lambda: tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_dssm,
    #                         var_list=variables_projections),
    #                            lambda : tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_dssm))

    # # dssm discriminator
    # query_product_dssm = tf.concat([query_out_2, product_out_2], 1)
    #
    # query_product_out_dssm = multiphi(name_training_domain, query_product_dssm, n_output=1, droupout_rate=droupout_rate,
    #                                   activation=None, name='query_product_dssm',
    #                                   isTraining=isTrain,
    #                                   dropout=True)
    #
    # y_prediction_classifier_dssm = query_product_out_dssm
    # cross_entropy_class_dssm = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction_classifier_dssm),
    #     name="cross_entropy_dssm")
    # pCTR_dssm = tf.nn.sigmoid(y_prediction_classifier_dssm, name='pCTR_dssm')
    # accuracy_dssm_domain, accuracy_dssm_domain_op = tf.metrics.accuracy(y_true,
    #                                                                     tf.cast(tf.greater_equal(pCTR_dssm, 0.5),
    #                                                                             tf.float32))
    # accuracy_dssm_domain_summary = tf.summary.scalar("Accuracy dssm domain classifier", accuracy_dssm_domain)
    # variables_classifier = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                          "query_product_dssm")
    # adam_train_step_class_dssm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_class_dssm,
    #                                                                                   var_list=variables_classifier)

    # GANS

    # GANS ON PRODUCTS!

    productname_triplets_target_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="productname_triplets_target_dssm")
    productname_triplets_emb_target_flat = tf.reshape(productname_triplets_target_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="productname_triplets_dssm_target_flat")
    product_target = tf.concat([productname_triplets_emb_target_flat], 1)
    product_out_target_1, _ = phi(product_target, n_output=second_layer_size_multi_domain,
                                  droupout_rate=droupout_rate,
                                  activation=None, name='generator_l0',
                                  isTraining=isTrain,
                                  dropout=False,
                                  reuse=tf.AUTO_REUSE)
    product_out_target_2, _ = phi(product_out_target_1, n_output=nb_triplets_query_product_buckets,
                                  droupout_rate=droupout_rate,
                                  activation=None, name='generator_l1',
                                  isTraining=isTrain,
                                  dropout=False,
                                  reuse=tf.AUTO_REUSE)
    y_prediction_discriminator_target, _ = phi(product_out_target_2, n_output=1,
                                               droupout_rate=droupout_rate,
                                               activation=None, name='product_discriminator_l2',
                                               isTraining=isTrain,
                                               dropout=False, reuse=tf.AUTO_REUSE)

    # y_prediction_discriminator_target = phi(product_out_target_2, n_output=1, droupout_rate=droupout_rate,
    #                                   activation=None, name='product_discriminator_l2',
    #                                   isTraining=isTrain,
    #                                   dropout=False,
    #                                   reuse=tf.AUTO_REUSE)

    productname_triplets_source_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="productname_triplets_source_dssm")
    productname_triplets_emb_source_flat = tf.reshape(productname_triplets_source_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="productname_triplets_dssm_source_flat")
    product_source = tf.concat([productname_triplets_emb_source_flat], 1)
    # product_out_source_1, _ =  phi(product_source, n_output=first_layer_size,
    #                          droupout_rate=droupout_rate,
    #                          activation=tf.nn.tanh, name='product_discriminator_l0',
    #                          isTraining=isTrain,
    #                          dropout=False,
    #                          reuse=tf.AUTO_REUSE)
    # product_out_source_2, _ =  phi(product_out_source_1, n_output=nb_triplets_query_product_buckets,
    #                          droupout_rate=droupout_rate,
    #                          activation=tf.nn.tanh, name='product_discriminator_l1',
    #                          isTraining=isTrain,
    #                          dropout=False,
    #                          reuse=tf.AUTO_REUSE)
    y_prediction_discriminator_source, _ = phi(product_source, n_output=1,
                                               droupout_rate=droupout_rate,
                                               activation=None, name='product_discriminator_l2',
                                               isTraining=isTrain,
                                               dropout=False,
                                               reuse=tf.AUTO_REUSE)

    # critics_loss_target = tf.reduce_mean(y_prediction_discriminator_target**2)
    # critics_loss_source = tf.reduce_mean((y_prediction_discriminator_source - 1) ** 2)
    # critics_loss = (critics_loss_target+critics_loss_source) / 2.0
    # generator_loss_wgans =  tf.reduce_mean((y_prediction_discriminator_target - 1) ** 2)
    #
    #
    # critics_loss_summary = tf.summary.scalar("Critics loss", -critics_loss)
    # critics_loss_summary_target = tf.summary.scalar("Critics loss target", critics_loss_target)
    # critics_loss_summary_source = tf.summary.scalar("Critics loss source", critics_loss_source)
    # generator_loss_summary_wass = tf.summary.scalar("Cross entropy generator wasserstein", generator_loss_wgans)

    y_true_source = tf.ones_like(y_prediction_discriminator_source)
    cross_entropy_discr_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true_source, logits=y_prediction_discriminator_source))
    cross_entropy_discr_source_summary = tf.summary.scalar("source discr entropy", cross_entropy_discr_source)
    y_true_target = tf.zeros_like(y_prediction_discriminator_target)
    cross_entropy_discr_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true_target, logits=y_prediction_discriminator_target))
    cross_entropy_discr_target_summary = tf.summary.scalar("target discr entropy", cross_entropy_discr_target)
    cross_entropy_discr = cross_entropy_discr_source + cross_entropy_discr_target
    cross_entropy_discr_summary = tf.summary.scalar("Cross entropy discr", cross_entropy_discr)
    cross_entropy_discr_source_summary = tf.summary.scalar("Cross entropy source", cross_entropy_discr_source)
    cross_entropy_discr_target_summary = tf.summary.scalar("Cross entropy target", cross_entropy_discr_target)

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(y_prediction_discriminator_target), logits=y_prediction_discriminator_target))

    generator_loss_summary = tf.summary.scalar("Cross entropy generator", generator_loss)

    variables_discr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "product_discriminator")
    variables_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    # clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for
    #                              var in variables_discr]

    optimize_critics = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='discr_optimizer').minimize(
        cross_entropy_discr,
        var_list=variables_discr)

    optimize_generator = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='gen_optimizer').minimize(
        generator_loss,
        var_list=variables_gen)

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
saver = tf.train.Saver(max_to_keep=None)
summary_op = tf.summary.merge_all()

modelname = sys.argv[1]
with tf.Session() as session:
    session.run(init)
    session.run(init_local)
    log_location = log_location + modelname
    print("Using log location for tensorboard {}".format(log_location))
    shutil.rmtree(log_location, ignore_errors=True)
    os.mkdir(log_location)
    summary_writer_gans = tf.summary.FileWriter(log_location + '/gans',
                                                session.graph)
    summary_writer_source_source = tf.summary.FileWriter(log_location + '/source_source',
                                                         session.graph)
    summary_writer_source_target = tf.summary.FileWriter(log_location + '/source_target',
                                                         session.graph)
    summary_writer_source_target_with_projection = tf.summary.FileWriter(
        log_location + '/source_target_with_projection',
        session.graph)
    summary_writer_target_target = tf.summary.FileWriter(log_location + '/target_target',
                                                         session.graph)
    step = 0


    for iteration_index in range(max_iterations):

        for batch_index in range(num_batches):
            step += 1

            # Training on source
            # batch_labels, query_productname_triplets_batch_data, query_triplets_batch_data, \
            # productname_triplets_batch_data = get_next_test_data(batch_positives, negative_samples_factor,
            #                                                      p_fptr, n_fptr)
            batch_labels, query_triplets_batch_data, productname_triplets_batch_data =\
                get_next_test_data_embeddings(batch_positives, negative_samples_factor,
                                          queries_source_embeddings_file,
                                          product_source_positive_embeddings_file,
                                          product_source_negative_embeddings_file)
            if batch_index % 10 == 0:

                # Source - Source domain validation
                # test_labels, test_query_productname_triplets, test_query_triplets, test_productname_triplets = \
                #     get_next_test_data(batch_positives * 5, 31, p_fptr, n_fptr)
                test_labels, test_query_triplets, test_productname_triplets = \
                    get_next_test_data_embeddings(batch_positives * 5, 31, queries_source_embeddings_file,
                                          product_source_positive_embeddings_file,
                                          product_source_negative_embeddings_file)
                fd_test = {y_true: test_labels,
                           query_triplets_emb: test_query_triplets,
                           productname_triplets_emb: test_productname_triplets,
                           isTrain: False,
                           name_training_domain: 'source',
                           isTargetProduct: False }
                val_loss_dssm, val_loss_dssm_summary = \
                   session.run([cross_entropy_dssm, cross_entropy_dssm_summary],feed_dict=fd_test)

                # summary_writer_source_source.add_summary(val_loss_summary, step)
                summary_writer_source_source.add_summary(val_loss_dssm_summary, step)

                if batch_index % 1000 == 0:
                    #print('iteration source-source letters ' + str(iteration_index) + ' batch ' + str(
                    #    batch_index + 1) + ' loss ' + str(
                    #    validation_loss) + ' done ')
                    print('iteration source-source dssm ' + str(iteration_index) + ' batch ' + str(
                        batch_index + 1) + ' loss ' + str(
                        val_loss_dssm) + ' done ')



            else:
                # Train on source domain
                session.run([adam_train_step_dssm], feed_dict={y_true: batch_labels,
                                                                                query_triplets_emb: query_triplets_batch_data,
                                                                                productname_triplets_emb: productname_triplets_batch_data,
                                                                                isTrain: True,
                                                                                name_training_domain: 'source',
                                                                                isTargetProduct: False})

queries_source_embeddings_file.close()
queries_target_embeddings_file.close()
product_source_positive_embeddings_file.close()
product_target_positive_embeddings_file.close()
product_source_negative_embeddings_file.close()
product_target_negative_embeddings_file.close()
