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
positives_training_file = path_data + "positive_training_samples_query_productname_stemmed_131.csv"
negatives_training_file = path_data + "negative_training_samples_query_productname_stemmed_131.csv"
test_positives_file = path_data + "test_positives_file_131"
test_negatives_file = path_data + "test_negatives_file_131"
positives_validation_file = path_data + "positive_training_samples_query_productname_stemmed_193.csv"
negatives_validation_file = path_data + "negative_training_samples_query_productname_stemmed_193.csv"
all_products_file = path_data + "negative_training_samples_query_productname_stemmed_notfrom_all.csv"
max_positive_training_samples_size = 11381259 - 381259
max_negative_training_samples_size = 27497519 - 497519
p_fptr = CircularFile(positives_training_file)
n_fptr = CircularFile(negatives_training_file)
fp_val = CircularFile(positives_validation_file)
fn_val = CircularFile(negatives_validation_file)

all_products = CircularFile(all_products_file)

# Parameteres

learning_rate = 5e-5
learning_rate_generator = 5e-5
droupout_rate = 0.4
first_layer_size = 300  # DSSM paper
second_layer_size = 100  # DSSM paper
batch_positives = 4
negative_samples_factor = 7
query_max_length = 25
productname_max_length = 90
brand_max_length = 25
nb_triplets_query_product_buckets = 2 ** 13
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


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def phi(x, n_output, droupout_rate, isTraining, name=None, batch_normalization=None, activation=None, reuse=None,
        dropout=None, orthogonal=None):
    n_input = x.get_shape().as_list()[1]
    bn_epsilon = 1e-3
    normal_axes = [0]

    with tf.variable_scope(name, reuse=reuse):
        if orthogonal:
            W = tf.get_variable(
                name='W',
                shape=[n_input, n_output],
                dtype=tf.float32,
                initializer=tf.orthogonal_initializer())
        else:
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
            gamma = tf.Variable(np.ones(n_output,np.float32),name='gamma')
            beta = tf.Variable(np.zeros(n_output,np.float32),name='beta')

            h = gamma * h + beta

        if activation:
            h = activation(h)

        if dropout:
            h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                        lambda: tf.layers.dropout(h, rate=0.0, training=True))

    return h


def Discriminator_Regularizer(D1_logits, D1_arg, D2_logits, D2_arg):
    D1 =  tf.reshape(tf.nn.sigmoid(D1_logits),[-1])
    D2 = tf.reshape(tf.nn.sigmoid(D2_logits),[-1])
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)
    grad_D1_logits_norm = tf.reshape(tf.norm(grad_D1_logits, axis=2, keep_dims=True),[-1])

    grad_D2_logits_norm =  tf.reshape(tf.norm(grad_D2_logits, axis=2, keep_dims=True), [-1])
    print('grad_D1_logits_norm.shape {} != D1.shape {}'.format(grad_D1_logits_norm.shape, D1.shape))
    print('grad_D2_logits_norm.shape {} != D2.shape {}'.format(grad_D2_logits_norm.shape, D2.shape))


    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    #assert grad_D1_logits_norm.shape == D1.shape
    #assert grad_D2_logits_norm.shape == D2.shape

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))

    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)

    return disc_regularizer



with tf.device('/cpu:0'):

    gamma_plh = tf.placeholder(tf.float32, shape=(), name='gamma')
    isTrain = tf.placeholder(tf.bool, shape=(), name="isTrain")
    training_mode = tf.placeholder(tf.string, shape=(), name="mode")

    #GANS ON PRODUCTS!

    productname_triplets_target_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="productname_triplets_target_dssm")
    productname_triplets_emb_target_flat = tf.reshape(productname_triplets_target_emb, [-1, nb_triplets_query_product_buckets],
                                               name="productname_triplets_dssm_target_flat")
    product_target = tf.concat([productname_triplets_emb_target_flat], 1)
    product_out_target_1 =  phi(product_target, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=leaky_relu, name='generator',
                             orthogonal=None,
                             isTraining=isTrain,
                             dropout=False,
                             reuse=tf.AUTO_REUSE)
    product_out_target_2 = phi(product_out_target_1, n_output=second_layer_size,
                             droupout_rate=droupout_rate,
                             activation=leaky_relu, name='product_discriminator_l1',
                             isTraining=isTrain,
                             dropout=False,reuse=tf.AUTO_REUSE)

    y_prediction_discriminator_target = phi(product_out_target_2, n_output=1, droupout_rate=droupout_rate,
                                      activation=None, name='product_discriminator_l2',
                                      isTraining=isTrain,
                                      dropout=False,
                                      reuse=tf.AUTO_REUSE)

    productname_triplets_source_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="productname_triplets_source_dssm")
    productname_triplets_emb_source_flat = tf.reshape(productname_triplets_source_emb, [-1, nb_triplets_query_product_buckets],
                                               name="productname_triplets_dssm_source_flat")
    product_source = tf.concat([productname_triplets_emb_source_flat], 1)
    product_out_source_1 =  phi(product_source, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=leaky_relu, name='product_discriminator_l0',
                             isTraining=isTrain,
                             dropout=False,
                             reuse=None,
                             orthogonal=True)
    product_out_source_2 = phi(product_out_source_1, n_output=second_layer_size,
                             droupout_rate=droupout_rate,
                             activation=leaky_relu, name='product_discriminator_l1',
                             isTraining=isTrain,
                             dropout=False,
                            reuse=tf.AUTO_REUSE)
    y_prediction_discriminator_source = phi(product_out_source_2, n_output=1, droupout_rate=droupout_rate,
                                      activation=None, name='product_discriminator_l2',
                                      isTraining=isTrain,
                                      dropout=False,
                                      reuse=tf.AUTO_REUSE)

    y_true_source = tf.ones_like(y_prediction_discriminator_source)
    cross_entropy_discr_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true_source, logits=y_prediction_discriminator_source))
    cross_entropy_discr_source_summary = tf.summary.scalar("source discr entropy",cross_entropy_discr_source)
    y_true_target = tf.zeros_like(y_prediction_discriminator_target)
    cross_entropy_discr_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true_target, logits=y_prediction_discriminator_target))
    cross_entropy_discr_target_summary = tf.summary.scalar("target discr entropy",cross_entropy_discr_target)
    cross_entropy_discr = cross_entropy_discr_source + cross_entropy_discr_target

    cross_entropy_discr_wgans = tf.reduce_mean(y_prediction_discriminator_target) - tf.reduce_mean(y_prediction_discriminator_source)

    #disc_reg = Discriminator_Regularizer(y_prediction_discriminator_source, product_source, y_prediction_discriminator_target, product_target)
    #disc_reg_summary = tf.summary.scalar('regularization',disc_reg)
    #cross_entropy_discr+=(gamma_plh/2.0)*disc_reg

    #print_op= tf.Print(cross_entropy_discr,[cross_entropy_discr],'output: ')

    cross_entropy_summary = tf.summary.scalar("Cross entropy  discr", cross_entropy_discr)
    cross_entropy_summary_wgans = tf.summary.scalar("Cross entropy  discr wasserstein", cross_entropy_discr_wgans)


    generator_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(y_prediction_discriminator_target), logits=y_prediction_discriminator_target))
    #generator_reg = tf.norm(product_out_source_2-product_out_target_2)/second_layer_size
    #generator_loss += 0.1*generator_reg

    generator_loss_wgans = -tf.reduce_mean(y_prediction_discriminator_target)

    generator_loss_summary = tf.summary.scalar("Cross entropy generator", generator_loss)
    generator_loss_summary_wass = tf.summary.scalar("Cross entropy generator wasserstein", generator_loss_wgans)



    variables_discr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "product_discriminator")
    variables_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for
                                 var in variables_discr]

    optimize_discr = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='discr_optimizer').minimize(cross_entropy_discr_wgans,
                                                                                      var_list=variables_discr)

    optimize_generator = tf.train.RMSPropOptimizer(learning_rate=learning_rate_generator, name='gen_optimizer').minimize(generator_loss_wgans,
                                                                                      var_list=variables_gen)

    pCTR_dssm = tf.concat([tf.nn.sigmoid(y_prediction_discriminator_source),
                           tf.nn.sigmoid(y_prediction_discriminator_target)],1)

    y_true = tf.concat([y_true_source,y_true_target],1)
    accuracy_discr, accuracy_dssm_domain_op = tf.metrics.accuracy(y_true,
                                                                  tf.cast(tf.greater_equal(pCTR_dssm, 0.5),
                                                                                tf.float32))
    accuracy_discr_summary = tf.summary.scalar("Accuracy discr", accuracy_discr)


    # # QUERY-PROUDUCT Prediction with Gans
    # y_true_dssm = tf.placeholder(tf.float32, shape=(None, 1))
    #
    # query_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
    #                                      name="query_triplets")
    # query_triplets_emb_flat = tf.reshape(query_triplets_emb, [-1, nb_triplets_query_product_buckets],
    #                                       name="query_triplets_flat")
    #
    # query_dssm_1 = phi(query_triplets_emb_flat, n_output=first_layer_size,
    #                        droupout_rate=droupout_rate,
    #                        activation=tf.nn.tanh, name='dssm_projection_gans_q_l1',
    #                        isTraining=isTrain,
    #                        dropout=None)
    # query_dssm_2 = phi(query_dssm_1, n_output=second_layer_size,
    #                        droupout_rate=droupout_rate,
    #                        activation=tf.nn.tanh, name='dssm_projection_gans_q_l2',
    #                        isTraining=isTrain,
    #                        dropout=None)
    # product_dssm_source_1 =  phi(product_target, n_output=first_layer_size,
    #                            droupout_rate=droupout_rate,
    #                            activation=tf.nn.tanh, name='generator',
    #                            batch_normalization=True,
    #                            isTraining=isTrain,
    #                            dropout=True,
    #                            reuse=tf.AUTO_REUSE)
    # product_dssm_source_2 = phi(product_dssm_source_1, n_output=second_layer_size,
    #                            droupout_rate=droupout_rate,
    #                            activation=tf.nn.tanh, name='dssm_projection_gans_p_l1',
    #                            isTraining=isTrain,
    #                            dropout=True,
    #                            reuse=None)
    # y_prediction_dssm = tf.reduce_sum(tf.multiply(query_dssm_2, product_dssm_source_2), 1, keep_dims=True)
    # cross_entropy_dssm = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_dssm, logits=y_prediction_dssm), name="cross_entropy_dssm")
    # cross_entropy_dssm_summary = tf.summary.scalar("Cross entropy DSSM", cross_entropy_dssm)
    # variables_dssm_projections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
    #                                          "dssm_projection_gans")
    # optimizer_dssm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_dssm,
    #                                                              var_list=variables_dssm_projections)


    # QUERY-PROUDUCT Prediction No Gans

    # query_dssm_1_nogans = phi(query_triplets_emb_flat, n_output=first_layer_size,
    #                        droupout_rate=droupout_rate,
    #                        activation=tf.nn.tanh, name='dssm_projection_nogans_q_l1',
    #                        isTraining=isTrain,
    #                        dropout=None,
    #                         batch_normalization=False)
    # query_dssm_2_nogans = phi(query_dssm_1_nogans, n_output=second_layer_size,
    #                        droupout_rate=droupout_rate,
    #                        activation=tf.nn.tanh, name='dssm_projection_nogans_q_l2',
    #                        isTraining=isTrain,
    #                        dropout=None,batch_normalization=False)
    #
    #
    #
    # product_dssm_source_1_nogans =  tf.cond(tf.equal(training_mode, tf.constant('nogans_projection')),
    #                           lambda: phi(product_source, n_output=first_layer_size,
    #                            droupout_rate=droupout_rate,
    #                            activation=tf.nn.tanh, name='generator_nogans',
    #                            batch_normalization=False,
    #                            isTraining=isTrain,
    #                            dropout=True),
    #                            lambda: phi(product_source, n_output=first_layer_size,
    #                            droupout_rate=droupout_rate,
    #                            activation=tf.nn.tanh, name='generator',
    #                            batch_normalization=False,
    #                            isTraining=isTrain,
    #                            dropout=True,
    #                            reuse=tf.AUTO_REUSE)
    #                                         )
    # product_dssm_source_2_nogans = phi(product_dssm_source_1_nogans, n_output=first_layer_size,
    #                            droupout_rate=droupout_rate,
    #                            activation=tf.nn.tanh, name='dssm_projection_nogans_p_l1',
    #                            isTraining=isTrain,
    #                            dropout=True,
    #                            reuse=None,batch_normalization=False)
    #
    # product_dssm_source_3_nogans = phi(product_dssm_source_2_nogans, n_output=second_layer_size,
    #                            droupout_rate=droupout_rate,
    #                            activation=tf.nn.tanh, name='dssm_projection_nogans_p_l2',
    #                            isTraining=isTrain,
    #                            dropout=True,
    #                            reuse=None,batch_normalization=False)
    # y_prediction_dssm_nogans = tf.reduce_sum(tf.multiply(query_dssm_2_nogans, product_dssm_source_3_nogans), 1, keep_dims=True)
    # cross_entropy_dssm_nogans = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_dssm, logits=y_prediction_dssm_nogans), name="cross_entropy_dssm")
    # cross_entropy_dssm_summary_nogans = tf.summary.scalar("Cross entropy DSSM", cross_entropy_dssm_nogans)
    # variables_dssm_projections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dssm_projection_nogans")
    # optimizer_dssm = tf.train.AdamOptimizer(learning_rate=learning_rate, name='dssm_optimizer').minimize(cross_entropy_dssm_nogans)

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
#saver = tf.train.Saver(max_to_keep=None)
summary_op = tf.summary.merge_all()

modelname = sys.argv[1]
max_iterations = 20
num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))

with tf.Session() as session:
    session.run(init)
    session.run(init_local)
    log_location = log_location + modelname
    print("Using log location for tensorboard {}".format(log_location))
    shutil.rmtree(log_location, ignore_errors=True)
    os.mkdir(log_location)
    summary_writer_gans = tf.summary.FileWriter(log_location + '/gans',
                                                         session.graph)
    summary_writer_nogans = tf.summary.FileWriter(log_location + '/nogans',
                                                         session.graph)

    step = 0

    for batch_index in range(30000000):
        step += 1
        # Training on source
        batch_labels_source, query_productname_triplets_batch_data, query_triplets_batch_data, \
        productname_triplets_batch_data = get_next_test_data(batch_positives, negative_samples_factor,
                                                             p_fptr, n_fptr)
        batch_labels_target, query_productname_triplets_batch_data_target, \
        query_triplets_batch_data_target, productname_triplets_batch_data_target \
            = get_next_test_data(batch_positives, negative_samples_factor, fp_val, fn_val)

        feed_dict = {
            productname_triplets_source_emb: productname_triplets_batch_data,
            productname_triplets_target_emb: productname_triplets_batch_data_target,
            isTrain: True, gamma_plh:0.001}

        if batch_index % 11 == 0:
            # gans validation
            feed_dict[isTrain] = False
            domain_acc_dssm_op = session.run(accuracy_dssm_domain_op, feed_dict)
            domain_acc_dssm, domain_acc_dssm_summary = session.run([accuracy_discr, accuracy_discr_summary])
            summary_writer_gans.add_summary(domain_acc_dssm_summary, step)
            gen_loss_v, gen_summary = session.run([generator_loss, generator_loss_summary], feed_dict)
            summary_writer_gans.add_summary(gen_summary, step)
            logloss, logloss_summary, logloss_source, logloss_target,\
                loglosswass, loglos_target_wass = session.run([cross_entropy_discr, cross_entropy_summary,
                                                    cross_entropy_discr_source_summary,cross_entropy_discr_target_summary,
                                                    cross_entropy_summary_wgans,generator_loss_summary_wass],
                                                   feed_dict)
            summary_writer_gans.add_summary(logloss_summary, step)
            summary_writer_gans.add_summary(logloss_source, step)
            summary_writer_gans.add_summary(logloss_target, step)
            summary_writer_gans.add_summary(loglosswass, step)
            summary_writer_gans.add_summary(loglos_target_wass, step)

            #summary_writer_gans.add_summary(reg, step)



        else:
            session.run([clip_discriminator_var_op])
            session.run([optimize_discr], feed_dict)

            if batch_index%5==0:
                session.run([optimize_generator], feed_dict)

            #if batch_index%5==0:


    # for iteration_index in range(max_iterations):
    #
    #     for batch_index in range(num_batches):
    #         step += 1
    #
    #         # Training on source
    #         batch_labels_source, query_productname_triplets_batch_data,query_triplets_batch_data,\
    #         productname_triplets_batch_data = get_next_test_data(batch_positives,negative_samples_factor,
    #                                                                                         p_fptr, n_fptr)
    #
    #         batch_labels_target, query_productname_triplets_batch_data_target, \
    #         query_triplets_batch_data_target, productname_triplets_batch_data_target \
    #             = get_next_test_data(batch_positives, negative_samples_factor, fp_val, fn_val)
    #
    #         feed_dict = {
    #                      productname_triplets_source_emb: productname_triplets_batch_data,
    #                      productname_triplets_target_emb: productname_triplets_batch_data_target,
    #                      query_triplets_emb:query_triplets_batch_data,
    #                      y_true_dssm : batch_labels_source,
    #                      isTrain: True,
    #                      training_mode: 'nogans_projection'}
    #         if batch_index % 10 == 0:
    #
    #             #dssm tested on target product
    #             feed_dict[isTrain] = False
    #             feed_dict[y_true_dssm]=batch_labels_target
    #             feed_dict[query_triplets_emb]=query_triplets_batch_data_target
    #             feed_dict[productname_triplets_source_emb]=productname_triplets_batch_data_target
    #             #feed_dict[productname_triplets_target_emb]=productname_triplets_batch_data_target
    #             #feed_dict[training_mode] = 'gans_projection'
    #             #logloss, logloss_summary = session.run([cross_entropy_dssm_nogans, cross_entropy_dssm_summary_nogans], feed_dict)
    #             #summary_writer_gans.add_summary(logloss_summary, step)
    #             feed_dict[training_mode] = 'nogans_projection'
    #             logloss, logloss_summary = session.run([cross_entropy_dssm_nogans, cross_entropy_dssm_summary_nogans], feed_dict)
    #             summary_writer_nogans.add_summary(logloss_summary, step)
    #
    #
    #         else:
    #             session.run(optimizer_dssm, feed_dict)






p_fptr.close()
n_fptr.close()
fp_val.close()
fn_val.close()

