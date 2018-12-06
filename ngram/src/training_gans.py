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



# cwd = '/var/opt/amin/Data/'
# log_location = cwd + 'logs/'
# path_data = cwd+'datasets/'
# path_model = cwd + 'models/'
positives_training_file = path_data + "positive_training_samples_query_productname_131.csv"
negatives_training_file = path_data + "negative_training_samples_query_productname_131.csv"

max_positive_training_samples_size = 11381259 - 381259
max_negative_training_samples_size = 27497519 - 497519
p_fptr = CircularFile(positives_training_file)
n_fptr = CircularFile(negatives_training_file)
fp_val = CircularFile(positives_validation_file)
fn_val = CircularFile(negatives_validation_file)

# Parameteres

learning_rate = 0.0001
droupout_rate = 0.4
first_layer_size = 300  # DSSM paper
second_layer_size = 100  # DSSM paper
# batch_positives = 4
# negative_samples_factor = 7
# query_max_length = 25
# productname_max_length = 90
# brand_max_length = 25
# nb_triplets_query_product_buckets = 2 ** 13
# batch_size = batch_positives * (negative_samples_factor + 1)
# miniBatchDisplayFreq = 100
# QUERY_INDEX = 0
# PRODCUTNAME_INDEX_POSITIVE = 1
# PRODCUTNAME_INDEX_NEGATIVE = 0
# PRODUCTDESC_INDEX_POSITIVE = 2
# PRODUCTDESC_INDEX_NEGATIVE = 1
# PRODUCTBRAND_INDEX_POSITIVE = 3
# PRODUCTBRAND_INDEX_NEGATIVE = 2
vocabulary_char_size = 76
num_filters_out = 256

product_filters = [4, 4, 3]
query_filters = [3, 4]

num_features_total = 20 * num_filters_out
char2Int={}
char2Int['a']=1
char2Int['b']= 2;
char2Int['c']= 3;
char2Int['d']= 4;
char2Int['e']= 5;
char2Int['f']= 6;
char2Int['g']= 7;
char2Int['h']= 8;
char2Int['i']= 9;
char2Int['j']= 10;
char2Int['k']= 11;
char2Int['l']= 12;
char2Int['m']=13;
char2Int['n']= 14;
char2Int['o']= 15;
char2Int['p']= 16;
char2Int['q']= 17;
char2Int['r']= 18;
char2Int['s']= 19;
char2Int['t']= 20;
char2Int['u']=21;
char2Int['v']= 22;
char2Int['w']= 23;
char2Int['x']= 24;
char2Int['y']= 25;
char2Int['z']= 26;
char2Int['0']= 27;
char2Int['1']= 28;
char2Int['2']= 29;
char2Int['3']= 30;
char2Int['4']= 31;
char2Int['5']= 32;
char2Int['6']= 33;
char2Int['7']= 34;
char2Int['8']= 35;
char2Int['9']= 36;

char2Int['(']= 37;
char2Int[')']= 38;
char2Int['<']= 39;
char2Int['>']= 40;
char2Int['{']= 41;
char2Int['}']= 42;
char2Int['[']= 43;
char2Int[']']= 44;

char2Int[' ']= 45;
char2Int['$']= 46;
char2Int['&']= 47;
char2Int[',']= 48;
char2Int[';']= 49;
char2Int['.']= 50;
char2Int['+']= 51;
char2Int['-']= 52;
char2Int['*']= 53;
char2Int['/']= 54;
char2Int['\\']= 55;
char2Int['|']= 56;
char2Int['%']= 57;
char2Int['^']= 58;
char2Int['=']= 59;
char2Int['~']= 60;
char2Int['_']= 61;
char2Int[':']= 62;
char2Int['@']= 63;
char2Int['!']= 64;
char2Int['#']= 65;
char2Int['?']= 66;
char2Int['"']= 67;
char2Int['\'']= 68;
char2Int['\u0080']= 69;
char2Int['\u0081']= 69;
char2Int['\u0082']= 69;
char2Int['\u0083']= 69;
char2Int['\u0084']= 69;
char2Int['\u0085']= 69;
char2Int['\u0086']= 69;
char2Int['\u0087']= 69;
char2Int['\u0088']= 69;
char2Int['\u0089']= 69;
char2Int['\u0090']= 70;
char2Int['\u0091']= 70;
char2Int['\u0092']= 70;
char2Int['\u0093']= 70;
char2Int['\u0094']= 70;
char2Int['\u0095']= 70;
char2Int['\u0096']= 70;
char2Int['\u0097']= 70;
char2Int['\u0098']= 70;
char2Int['\u0099']= 70;
char2Int['\u00a0']= 71;
char2Int['\u00a1']= 71;
char2Int['\u00a2']= 71;
char2Int['\u00a3']= 71;
char2Int['\u00a4']= 71;
char2Int['\u00a5']= 71;
char2Int['\u00a6']=71;
char2Int['\u00a7']= 71;
char2Int['\u00a8']= 71;
char2Int['\u00a9']= 71;
char2Int['\u00b0']= 72;
char2Int['\u00b1']= 72;
char2Int['\u00b2']= 72;
char2Int['\u00b3']= 72;
char2Int['\u00b4']= 72;
char2Int['\u00b5']= 72;
char2Int['\u00b6']= 72;
char2Int['\u00b7']= 72;
char2Int['\u00b8']= 72;
char2Int['\u00b9']= 72;

char2Int['\u00c0']= 73;
char2Int['\u00c1']= 73;
char2Int['\u00c2']= 73;
char2Int['\u00c3']= 73;
char2Int['\u00c4']= 73;
char2Int['\u00c5']= 73;
char2Int['\u00c6']= 73;
char2Int['\u00c7']= 73;
char2Int['\u00c8']=73;
char2Int['\u00c9']= 73;

char2Int['\u00d0']= 74;
char2Int['\u00d1']= 74;
char2Int['\u00d2']= 74;
char2Int['\u00d3']= 74;
char2Int['\u00d4']= 74;
char2Int['\u00d5']= 74;
char2Int['\u00d6']= 74;
char2Int['\u00d7']= 74;
char2Int['\u00d8']= 74;
char2Int['\u00d9']= 74;
char2Int['\u00e0']= 75;
char2Int['\u00e1']= 75;
char2Int['\u00e2']= 75;
char2Int['\u00e3']= 75;
char2Int['\u00e4']= 75;
char2Int['\u00e5']= 75;
char2Int['\u00e6']= 75;
char2Int['\u00e7']= 75;
char2Int['\u00e8']= 75;
char2Int['\u00e9']= 75;




def getIntForChar(c):
    if c in char2Int:
        return char2Int[c]
    return vocabulary_char_size-1

with open(positives_training_file) as f:
    max_positive_training_samples_size=len(f.readlines())

p_fptr = CircularFile(positives_training_file)
n_fptr = CircularFile(negatives_training_file)


learning_rate = 0.0001
droupout_rate = 0.4
batch_positives = 4
negative_samples_factor = 7
query_max_length = 25
productname_max_length = 90
brand_max_length = 25
nb_triplets_query_product_buckets = 2**15
batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 100

QUERY_INDEX = 0
PRODCUTNAME_INDEX_POSITIVE = 1
PRODUCTBRAND_INDEX_POSITIVE = 3
PRODCUTNAME_INDEX_NEGATIVE = 0
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

def product2tensor(product):
    product_norm = normalizeProduct(product,productname_max_length)
    tensor_index=[]
    for c in product_norm:
        tensor_index.append(getIntForChar(c))
    return tensor_index

def query2tensor(query):
    query_norm = normalizeQuery(query,query_max_length)
    tensor_index=[]
    for c in query_norm:
        tensor_index.append(getIntForChar(c))
    return tensor_index


def brand2tensor(brand):
    brand_norm = normalizeProduct(brand,productname_max_length)
    tensor_index=[]
    for c in brand_norm:
        tensor_index.append(getIntForChar(c))
    return tensor_index



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

    #character level input
    #query_char_data = np.zeros((test_data_size, query_max_length, vocabulary_char_size + 1, 1))
    #product_char_data = np.zeros((test_data_size, productname_max_length, vocabulary_char_size + 1, 1))


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


        #deepmatch

        #query_chars = query2tensor(querynormalized)
        #product_chars = product2tensor(productnamenormalized)
        #for cidx in range(len(query_chars)):
        #    query_char_data[product_index, cidx, int(query_chars[cidx]), 0] = 1
        #for cidx in range(len(product_chars)):
        #    product_char_data[product_index, cidx, int(product_chars[cidx]), 0] = 1


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

            #query_chars = query2tensor(ptokens[QUERY_INDEX])
            #product_chars = product2tensor(productnamenormalized)
            #for cidx in range(len(query_chars)):
            #    query_char_data[product_index, cidx, int(query_chars[cidx]), 0] = 1
            #for cidx in range(len(product_chars)):
            #    product_char_data[product_index, cidx, int(product_chars[cidx]), 0] = 1

            product_index += 1
            negatives += 1
    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index, 0] = 1
    return test_labels, query_productname_triplets, query_triplets, productname_triplets
    #query_char_data, product_char_data



def get_next_test_data_char(positives_count, test_negative_samples_factor, fpp, fnn):
    # test_negative_samples_factor = 1  # negative_samples_factor
    test_data_size = positives_count * (test_negative_samples_factor + 1)
    test_labels = np.zeros((test_data_size, 1))



    #character level input
    query_char_data = np.zeros((test_data_size, query_max_length, vocabulary_char_size + 1, 1))
    product_char_data = np.zeros((test_data_size, productname_max_length, vocabulary_char_size + 1, 1))


    # query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    # query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    product_index = 0

    for index in range(positives_count):
        positives_line = fpp.readline()
        ptokens = positives_line.rstrip().split('|')
        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;

        #deepmatch

        query_chars = query2tensor(ptokens[QUERY_INDEX])
        product_chars = product2tensor(ptokens[PRODCUTNAME_INDEX_POSITIVE])
        for cidx in range(len(query_chars)):
            query_char_data[product_index, cidx, int(query_chars[cidx]), 0] = 1
        for cidx in range(len(product_chars)):
            product_char_data[product_index, cidx, int(product_chars[cidx]), 0] = 1


        product_index += 1
        negatives = 0

        while (negatives != test_negative_samples_factor):
            negatives_line = fnn.readline()
            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;

            #query_chars = query2tensor(ptokens[QUERY_INDEX])
            product_chars = product2tensor(ntokens[PRODCUTNAME_INDEX_NEGATIVE])
            for cidx in range(len(query_chars)):
                query_char_data[product_index, cidx, int(query_chars[cidx]), 0] = 1
            for cidx in range(len(product_chars)):
                product_char_data[product_index, cidx, int(product_chars[cidx]), 0] = 1

            product_index += 1
            negatives += 1
    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index, 0] = 1
    return test_labels, query_char_data, product_char_data



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
            gamma = tf.Variable(np.ones(n_output,np.float32),name='gamma')
            beta = tf.Variable(np.zeros(n_output,np.float32),name='beta')

            h = gamma * h + beta

        if activation:
            h = activation(h)

        if dropout:
            h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                        lambda: tf.layers.dropout(h, rate=0.0, training=True))

    return h, W

def psi(x, filter_height, filter_width, n_filters_in, n_filters_out, conv_strides, conv_padding,
        droupout_rate, isTraining, name=None, activation=None, reuse=None, batch_normalization=None,
        dropout=None):
    bn_epsilon = 1e-3
    conv_axes = [0, 1, 2]

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[filter_height, filter_width, n_filters_in, n_filters_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_filters_out],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(tf.nn.conv2d(input=x,
                 filter=W,
                 strides=conv_strides,
                 padding=conv_padding),
            b)


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
                                                lambda:  tf.cond(tf.equal(parameter_group, tf.constant('source_pr')),
                                                                 lambda: phi(input, n_output=n_output, droupout_rate=droupout_rate,
                                                                                activation=activation,
                                                                                name=name+'source_pr',
                                                                                batch_normalization=batch_normalization,
                                                                                isTraining=isTraining,
                                                                                dropout=dropout, reuse=reuse),
                                                                 lambda: phi(input, n_output=n_output,
                                                                             droupout_rate=droupout_rate,
                                                                             activation=activation,
                                                                             name=name+'discriminator',
                                                                             batch_normalization=batch_normalization,
                                                                             isTraining=isTraining,
                                                                             dropout=dropout, reuse=reuse)
                                                                 )
                                                )
                                )
                )
    return query_product_out_2


with tf.device('/cpu:0'):




    # query-product classifier
    max_iterations = 20
    num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))
    isTrain = tf.placeholder(tf.bool, shape=(), name="isTrain")
    isTargetProduct = tf.placeholder(tf.bool, shape=(), name="isTargetProduct")
    name_training_domain = tf.placeholder(tf.string, shape=(), name="domain")


    # Letter nGram Model LR
    query_productname_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                    name="query_productname_triplets")
    query_productname_triplets_emb_flat = tf.reshape(query_productname_triplets_emb,
                                                     [-1, nb_triplets_query_product_buckets],
                                                     name="query_productname_triplets_flat")

    query_product = tf.concat([query_productname_triplets_emb_flat], 1)
    y_true = tf.placeholder(tf.float32, shape=(None, 1))

    query_product_out_2 = multiphi(name_training_domain, query_product, n_output=1, droupout_rate=droupout_rate,
                                   activation=None, name='query_product_out_2',
                                   isTraining=isTrain,
                                   dropout=True)

    y_prediction = query_product_out_2
    pCTR = tf.nn.sigmoid(y_prediction, name="pCTR")
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction))
    cross_entropy_summary = tf.summary.scalar("Cross entropy", cross_entropy)
    accuracy_domain, accuracy_domain_op = tf.metrics.accuracy(y_true, tf.cast(tf.greater_equal(pCTR, 0.5), tf.float32))
    accuracy_domain_summary = tf.summary.scalar("Accuracy domain classifier", accuracy_domain)
    adam_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # projection of target product
    #productname_target_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
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

    # query_out_1 = multiphi(name_training_domain, query_triplets_emb_flat, n_output=first_layer_size,
    #                        droupout_rate=droupout_rate,
    #                        activation=tf.nn.tanh, name='qp_triplets_emb_projection_q1',
    #                        isTraining=isTrain,
    #                        dropout=True)

    query_out_1 =  tf.cond(isTargetProduct,
                             lambda: tf.cond(isTrain,
                                         lambda: multiphi("", query_triplets_emb_flat, n_output=first_layer_size,
                                         droupout_rate=droupout_rate,
                                         activation=tf.nn.tanh, name='product_discriminator_l0',
                                         batch_normalization=True,
                                         isTraining=isTrain,
                                         dropout=False,
                                         reuse=tf.AUTO_REUSE),
                                         lambda: multiphi("", query_triplets_emb_flat, n_output=first_layer_size,
                                          droupout_rate=droupout_rate,
                                          activation=tf.nn.tanh, name='generator',
                                          batch_normalization=True,
                                          isTraining=isTrain,
                                          dropout=False,
                                          reuse=tf.AUTO_REUSE)
                              ),
                             lambda: multiphi(name_training_domain, query_triplets_emb_flat, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='qp_triplets_emb_projection_q1',
                             batch_normalization=True,
                             isTraining=isTrain,
                             dropout=True)
                             )

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
    product_out_1 =  tf.cond(isTargetProduct,
                             lambda: tf.cond(isTrain,
                                         lambda: multiphi("", product, n_output=first_layer_size,
                                         droupout_rate=droupout_rate,
                                         activation=tf.nn.tanh, name='product_discriminator_l0',
                                         batch_normalization=True,
                                         isTraining=isTrain,
                                         dropout=False,
                                         reuse=tf.AUTO_REUSE),
                                         lambda: multiphi("", product, n_output=first_layer_size,
                                          droupout_rate=droupout_rate,
                                          activation=tf.nn.tanh, name='generator',
                                          batch_normalization=True,
                                          isTraining=isTrain,
                                          dropout=False,
                                          reuse=tf.AUTO_REUSE)
                              ),
                             lambda: multiphi(name_training_domain, product, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='qp_triplets_emb_projection_p1',
                             batch_normalization=True,
                             isTraining=isTrain,
                             dropout=True)
                             )

    product_out_2 = multiphi(name_training_domain, product_out_1, n_output=second_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='qp_triplets_emb_projection_p2',
                             isTraining=isTrain,
                             dropout=True)
    # y_true = tf.placeholder(tf.float32, shape=(None, 1))
    y_prediction_dssm = tf.reduce_sum(tf.multiply(query_out_2, product_out_2), 1, keep_dims=True)
    cross_entropy_dssm = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction_dssm), name="cross_entropy_dssm")
    cross_entropy_dssm_summary = tf.summary.scalar("Cross entropy DSSM", cross_entropy_dssm)
    variables_projections = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "qp_triplets_emb_projection")
    adam_train_step_dssm = tf.cond(isTargetProduct,
                                   lambda: tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_dssm,
                                var_list=variables_projections),
                                   lambda : tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_dssm))

    # dssm discriminator
    query_product_dssm = tf.concat([query_out_2, product_out_2], 1)

    query_product_out_dssm = multiphi(name_training_domain, query_product_dssm, n_output=1, droupout_rate=droupout_rate,
                                      activation=None, name='query_product_dssm',
                                      isTraining=isTrain,
                                      dropout=True)

    y_prediction_classifier_dssm = query_product_out_dssm
    cross_entropy_class_dssm = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction_classifier_dssm),
        name="cross_entropy_dssm")
    pCTR_dssm = tf.nn.sigmoid(y_prediction_classifier_dssm, name='pCTR_dssm')
    accuracy_dssm_domain, accuracy_dssm_domain_op = tf.metrics.accuracy(y_true,
                                                                        tf.cast(tf.greater_equal(pCTR_dssm, 0.5),
                                                                                tf.float32))
    accuracy_dssm_domain_summary = tf.summary.scalar("Accuracy dssm domain classifier", accuracy_dssm_domain)
    variables_classifier = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "query_product_dssm")
    adam_train_step_class_dssm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_class_dssm,
                                                                                      var_list=variables_classifier)


    # deep match

    # product_input = tf.placeholder(tf.float32, [None, productname_max_length, vocabulary_char_size + 1, 1], name="product")
    # tf.add_to_collection("product_input", product_input)
    #
    # product_out_1, product_wt_1 = psi(product_input, filter_height=product_filters[0], filter_width=vocabulary_char_size + 1,
    #                                   n_filters_in=1, n_filters_out=256,
    #                                   conv_strides=[1, 1, 1, 1], conv_padding='VALID', activation=tf.nn.relu,
    #                                   name='product_conv_layer_1', droupout_rate=droupout_rate, isTraining=isTrain,
    #                                dropout=True)
    #
    # product_out_2, product_wt_2 = psi(product_out_1, filter_height=product_filters[1], filter_width=1,
    #                                   n_filters_in=256, n_filters_out=256,
    #                                   conv_strides=[1, 1, 1, 1], conv_padding='VALID', activation=tf.nn.relu,
    #                                   name='product_conv_layer_2', droupout_rate=droupout_rate,  isTraining=isTrain,
    #                                dropout=True)
    #
    # product_out_3 = tf.nn.max_pool(product_out_2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID',
    #                                name="product_pool_1")
    #
    # product_out_4, product_wt_3 = psi(product_out_3, filter_height=product_filters[2], filter_width=1,
    #                                   n_filters_in=256, n_filters_out=256,
    #                                   conv_strides=[1, 1, 1, 1], conv_padding='VALID', activation=tf.nn.relu,
    #                                   name='product_conv_layer_3', droupout_rate=droupout_rate, isTraining=isTrain,
    #                                dropout=True)
    #
    # product_out_5 = tf.nn.max_pool(product_out_4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID',
    #                                name="product_pool_2")
    #
    # product_flat = tf.reshape(product_out_5, [-1, num_features_total], name="product_flat")
    # tf.add_to_collection("product_flat", product_flat)
    #
    # # product block end
    #
    # # query block begin
    #
    # query_input = tf.placeholder(tf.float32, [None, query_max_length, vocabulary_char_size + 1, 1], name="query")
    # tf.add_to_collection("query_input", query_input)
    #
    #
    # query_out_1, query_wt_1 = psi(query_input, filter_height=query_filters[0], filter_width=vocabulary_char_size + 1,
    #                               n_filters_in=1, n_filters_out=256,
    #                               conv_strides=[1, 1, 1, 1], conv_padding='VALID', activation=tf.nn.relu,
    #                               name='query_conv_layer_1', droupout_rate=droupout_rate, isTraining=isTrain,
    #                                dropout=True)
    #
    # query_out_2, query_wt_2 = psi(query_out_1, filter_height=query_filters[1], filter_width=1,
    #                               n_filters_in=256, n_filters_out=256,
    #                               conv_strides=[1, 1, 1, 1], conv_padding='VALID', activation=tf.nn.relu,
    #                               name='query_conv_layer_2',  droupout_rate=droupout_rate, isTraining=isTrain,
    #                                dropout=True)
    #
    # query_flat = tf.reshape(query_out_2, [-1, num_features_total], name="query_flat")
    #
    # query_product = tf.concat([query_flat, product_flat],1)
    # query_product_out_1 , query_product_out_wt_1 = phi(query_product,n_output=256, droupout_rate=droupout_rate, activation=tf.nn.relu,  name='query_fc_layer_1',
    #                 isTraining=isTrain,
    #                 batch_normalization = False, dropout=True)
    #
    # query_product_out_2 , query_product_out_wt_2 = phi(query_product_out_1,n_output=1,  droupout_rate=droupout_rate, activation=None, name='query_fc_layer_2',
    #                 isTraining=isTrain,
    #                 batch_normalization = False, dropout=False)
    #
    #
    # y_prediction_deepmatch = query_product_out_2
    # pCTR_dssm = tf.nn.sigmoid(y_prediction_deepmatch, name='pCTR_deepMatch')
    #
    # cross_entropy_deep_match = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction_deepmatch),
    #                                           name='cross_entropy_deep_match')
    # cross_entropy_deep_match_summary = tf.summary.scalar("Cross entropy Deep Match", cross_entropy_deep_match)
    #
    # adam_train_step_deep_match = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_deep_match)

    #GANS

    #GANS ON PRODUCTS!

    productname_triplets_target_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="productname_triplets_target_dssm")
    productname_triplets_emb_target_flat = tf.reshape(productname_triplets_target_emb, [-1, nb_triplets_query_product_buckets],
                                               name="productname_triplets_dssm_target_flat")
    product_target = tf.concat([productname_triplets_emb_target_flat], 1)
    product_out_target_1, _ =  phi(product_target, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='generator',
                             isTraining=isTrain,
                             dropout=False,
                             reuse=tf.AUTO_REUSE)
    y_prediction_discriminator_target, _ = phi(product_out_target_1, n_output=1,
                             droupout_rate=droupout_rate,
                             activation=None, name='product_discriminator_l1',
                             isTraining=isTrain,
                             dropout=False,reuse=tf.AUTO_REUSE)

    # y_prediction_discriminator_target = phi(product_out_target_2, n_output=1, droupout_rate=droupout_rate,
    #                                   activation=None, name='product_discriminator_l2',
    #                                   isTraining=isTrain,
    #                                   dropout=False,
    #                                   reuse=tf.AUTO_REUSE)

    productname_triplets_source_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="productname_triplets_source_dssm")
    productname_triplets_emb_source_flat = tf.reshape(productname_triplets_source_emb, [-1, nb_triplets_query_product_buckets],
                                               name="productname_triplets_dssm_source_flat")
    product_source = tf.concat([productname_triplets_emb_source_flat], 1)
    product_out_source_1, _ =  phi(product_source, n_output=first_layer_size,
                             droupout_rate=droupout_rate,
                             activation=tf.nn.tanh, name='product_discriminator_l0',
                             isTraining=isTrain,
                             dropout=False,
                             reuse=tf.AUTO_REUSE)
    y_prediction_discriminator_source, _ = phi(product_out_source_1, n_output=1,
                             droupout_rate=droupout_rate,
                             activation=None, name='product_discriminator_l1',
                             isTraining=isTrain,
                             dropout=False,
                            reuse=tf.AUTO_REUSE)



    critics_loss_target = tf.reduce_mean(y_prediction_discriminator_target**2)
    critics_loss_source = tf.reduce_mean((y_prediction_discriminator_source - 1) ** 2)
    critics_loss = (critics_loss_target+critics_loss_source) / 2.0
    generator_loss_wgans =  tf.reduce_mean((y_prediction_discriminator_target - 1) ** 2)


    critics_loss_summary = tf.summary.scalar("Critics loss", -critics_loss)
    critics_loss_summary_target = tf.summary.scalar("Critics loss target", critics_loss_target)
    critics_loss_summary_source = tf.summary.scalar("Critics loss source", critics_loss_source)
    generator_loss_summary_wass = tf.summary.scalar("Cross entropy generator wasserstein", generator_loss_wgans)





    variables_discr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "product_discriminator")
    variables_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

    clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for
                                 var in variables_discr]

    optimize_critics = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='discr_optimizer').minimize(critics_loss,
                                                                                                               var_list=variables_discr)

    optimize_generator = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='gen_optimizer').minimize(generator_loss_wgans,
                                                                                      var_list=variables_gen)


init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)
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
    summary_writer_source_target_with_projection = tf.summary.FileWriter(log_location + '/source_target_with_projection',
                                                         session.graph)
    summary_writer_target_target = tf.summary.FileWriter(log_location + '/target_target',
                                                         session.graph)
    step = 0

    for batch_index in range(8000):
        step += 1
        # Training on source
        batch_labels_source, query_productname_triplets_batch_data, query_triplets_batch_data, \
        productname_triplets_batch_data = get_next_test_data(batch_positives, negative_samples_factor,p_fptr, n_fptr)

        batch_labels_target, query_productname_triplets_batch_data_target, \
        query_triplets_batch_data_target, \
        productname_triplets_batch_data_target = get_next_test_data(batch_positives, negative_samples_factor, fp_val, fn_val)

        feed_dict = {
            productname_triplets_source_emb: productname_triplets_batch_data,
            productname_triplets_target_emb: productname_triplets_batch_data_target,
            isTrain: True}

        if batch_index % 11 == 0:
            # gans validation
            feed_dict[isTrain] = False
            # domain_acc_dssm_op = session.run(accuracy_dssm_domain_op, feed_dict)
            # domain_acc_dssm, domain_acc_dssm_summary = session.run([accuracy_discr, accuracy_discr_summary])
            gen_loss_v, gen_summary = session.run([generator_loss_wgans, generator_loss_summary_wass], feed_dict)
            summary_writer_gans.add_summary(gen_summary, step)
            logloss, logloss_summary, logloss_source, logloss_target  = session.run([critics_loss, critics_loss_summary,
                                                               critics_loss_summary_source, critics_loss_summary_target
                                                                ],
                                                              feed_dict)
            summary_writer_gans.add_summary(logloss_summary, step)
            summary_writer_gans.add_summary(logloss_source, step)
            summary_writer_gans.add_summary(logloss_target, step)

        else:
            session.run([optimize_critics], feed_dict)
            session.run([optimize_generator], feed_dict)



    for iteration_index in range(max_iterations):

        for batch_index in range(num_batches):
            step += 1

            # Training on source
            batch_labels, query_productname_triplets_batch_data,query_triplets_batch_data,\
            productname_triplets_batch_data \
                    = get_next_test_data(batch_positives,negative_samples_factor, p_fptr, n_fptr)

            # batch_labels, query_char_batch_data, product_char_batch_data \
            #         = get_next_test_data_char(batch_positives,negative_samples_factor, p_fptr, n_fptr)

            if batch_index % 10 == 0:

                # Source - Source domain validation
                test_labels, test_query_productname_triplets, test_query_triplets, test_productname_triplets = \
                    get_next_test_data(batch_positives * 5, 31, p_fptr, n_fptr)

               # test_labels, test_query_char_batch_data, test_product_char_batch_data= \
               #      get_next_test_data_char(batch_positives * 5, 31, p_fptr, n_fptr)


                fd_test = {y_true: test_labels,
                           query_productname_triplets_emb: test_query_productname_triplets,
                           query_triplets_emb: test_query_triplets,
                           productname_triplets_emb: test_productname_triplets,
                           isTrain: False, name_training_domain: 'source',isTargetProduct: False
                           }
                           # query_input: test_query_char_batch_data, product_input: test_product_char_batch_data
                           #}
                validation_loss, val_loss_summary, val_loss_dssm, val_loss_dssm_summary\
                    = session.run(
                    [cross_entropy, cross_entropy_summary, cross_entropy_dssm, cross_entropy_dssm_summary],
                    feed_dict=fd_test)

                # validation_loss_deep_match, validation_loss_deep_match_summary = session.run(
                #     [cross_entropy_deep_match, cross_entropy_deep_match_summary],
                #     feed_dict=fd_test)
                summary_writer_source_source.add_summary(val_loss_summary, step)
                summary_writer_source_source.add_summary(val_loss_dssm_summary, step)
                #summary_writer_source_source.add_summary(validation_loss_deep_match_summary, step)

                if batch_index % 1000 == 0:
                #     print('iteration source-source letters ' + str(iteration_index) + ' batch ' + str(
                #         batch_index + 1) + ' loss ' + str(
                #         validation_loss) + ' done ')
                #     print('iteration source-source dssm ' + str(iteration_index) + ' batch ' + str(
                #         batch_index + 1) + ' loss ' + str(
                #         val_loss_dssm) + ' done ')
                     saver.save(session, path_model+modelname,global_step=step)


                # Source - Target domain validation
                test_labels, test_query_productname_triplets, test_query_triplets, test_productname_triplets = \
                    get_next_test_data(batch_positives * 5, 31, fp_val, fn_val)
                fd_test = {y_true: test_labels, query_productname_triplets_emb: test_query_productname_triplets,
                           query_triplets_emb: test_query_triplets,
                           productname_triplets_emb: test_productname_triplets,
                           isTrain: False, name_training_domain: 'source',isTargetProduct: False}
                val_loss_summary, val_loss_dssm_summary = session.run(
                    [cross_entropy_summary, cross_entropy_dssm_summary],
                    feed_dict=fd_test)
                summary_writer_source_target.add_summary(val_loss_summary, step)
                summary_writer_source_target.add_summary(val_loss_dssm_summary, step)
                if batch_index % 1000 == 0:
                    print( 'iteration source-target ' + str(iteration_index) + ' batch ' + str(batch_index + 1) + ' loss '
                        + str(validation_loss) + ' done ')

                # # Source - Target domain validation: using projected products models

                fd_test[name_training_domain]='source_pr'
                fd_test[isTargetProduct]=True ##use generator projection
                val_loss_dssm_summary = session.run(
                    cross_entropy_dssm_summary,
                    feed_dict=fd_test)
                summary_writer_source_target_with_projection.add_summary(val_loss_dssm_summary, step)
                if batch_index % 1000 == 0:
                    print( 'iteration source-target with projected products' + str(iteration_index) + ' batch ' + str(batch_index + 1) + ' loss '
                        + str(validation_loss) + ' done ')

                #
                # Target - Target domain validation
                fd_test[name_training_domain]='target'
                fd_test[isTargetProduct]=False
                validation_loss, val_loss_summary, val_loss_dssm, val_loss_dssm_summary = session.run(
                    [cross_entropy, cross_entropy_summary, cross_entropy_dssm, cross_entropy_dssm_summary],
                    feed_dict=fd_test)
                summary_writer_target_target.add_summary(val_loss_summary, step)
                summary_writer_target_target.add_summary(val_loss_dssm_summary, step)
                if batch_index % 1000 == 0:
                    print('iteration target-target ' + str(iteration_index) + ' batch ' + str(batch_index + 1) + ' loss '
                        + str(validation_loss) + ' done ')
                    print('iteration target-target ' + str(iteration_index) + ' batch ' + str(batch_index + 1) + ' loss '
                        + str(val_loss_dssm) + ' done ')


            else:
                # Train on source domain
                session.run([adam_train_step, adam_train_step_dssm], feed_dict={y_true: batch_labels,
                                                                                query_productname_triplets_emb: query_productname_triplets_batch_data,
                                                                                query_triplets_emb: query_triplets_batch_data,
                                                                                productname_triplets_emb: productname_triplets_batch_data,
                                                                                isTrain: True,
                                                                                name_training_domain: 'source', isTargetProduct: False,
                                                                                #query_input: query_char_batch_data,
                                                                                #product_input: product_char_batch_data
                                                                                 })

                # session.run([adam_train_step, adam_train_step_dssm], feed_dict={y_true: batch_labels,
                #                                                                 query_productname_triplets_emb: query_productname_triplets_batch_data,
                #                                                                 query_triplets_emb: query_triplets_batch_data,
                #                                                                 productname_triplets_emb: productname_triplets_batch_data,
                #                                                                 isTrain: True,
                #                                                                 name_training_domain: 'source_pr', isTargetProduct: True})

                # session.run([adam_train_step_dssm], feed_dict={y_true: batch_labels,
                #                                                                 query_productname_triplets_emb: query_productname_triplets_batch_data,
                #                                                                 query_triplets_emb: query_triplets_batch_data,
                #                                                                 productname_triplets_emb: productname_triplets_batch_data,
                #                                                                 isTrain: True,
                #                                                                 name_training_domain: 'source_projectedproducts', isTargetProduct: False})


                # Train on positive from query and product positive from source, negative products from target
                # batch_labels, query_productname_triplets_batch_data_target, \
                # query_triplets_batch_data_target, productname_triplets_batch_data_target \
                #     = get_next_test_data(batch_positives, negative_samples_factor, p_fptr, fn_val)
                # session.run([adam_train_step_dssm], feed_dict={y_true: batch_labels,
                #                                                                 query_productname_triplets_emb: query_productname_triplets_batch_data_target,
                #                                                                 query_triplets_emb: query_triplets_batch_data_target,
                #                                                                 productname_triplets_emb: productname_triplets_batch_data_target,
                #                                                                 isTrain: True,
                #                                                                 name_training_domain: 'source_projectedproducts',isTargetProduct: True})
                #

                # Train on target domain
                batch_labels, query_productname_triplets_batch_data_target, \
                query_triplets_batch_data_target, productname_triplets_batch_data_target \
                    = get_next_test_data(batch_positives, negative_samples_factor, fp_val, fn_val)
                session.run([adam_train_step, adam_train_step_dssm], feed_dict={y_true: batch_labels,
                                                                                query_productname_triplets_emb: query_productname_triplets_batch_data_target,
                                                                                query_triplets_emb: query_triplets_batch_data_target,
                                                                                productname_triplets_emb: productname_triplets_batch_data_target,
                                                                                isTrain: True,
                                                                                name_training_domain: 'target',isTargetProduct: False})

                # # Discriminator
                # indices = np.random.permutation(range(0, len(query_productname_triplets_batch_data_target) +
                #                                       len(query_productname_triplets_batch_data)))
                # domain_labels = np.concatenate((np.zeros((len(query_productname_triplets_batch_data), 1)),
                #                                 np.ones((len(query_productname_triplets_batch_data_target), 1))))
                # query_productname_triplets_batch_data = np.take(np.concatenate(
                #     (query_productname_triplets_batch_data, query_productname_triplets_batch_data_target)), indices, 0)
                # query_triplets_batch_data = np.take(np.concatenate((query_triplets_batch_data, query_triplets_batch_data_target)),
                #                                     indices, 0)
                # productname_triplets_batch_data = np.take(np.concatenate((productname_triplets_batch_data, productname_triplets_batch_data_target)),
                #                                           indices,0)
                # domain_labels = np.take(domain_labels, indices, 0)
                # feed_dict = {y_true: domain_labels,
                #              query_productname_triplets_emb: query_productname_triplets_batch_data,
                #              query_triplets_emb: query_triplets_batch_data,
                #              productname_triplets_emb: productname_triplets_batch_data,
                #              isTrain: True, name_training_domain: 'domain_classifier_LR', isTargetProduct: False}

                # if batch_index % 11 == 0:
                #     # Validation of discriminators
                #     # logistic discriminator
                #     feed_dict[isTrain] = False
                #     domain_acc_op = session.run(accuracy_domain_op, feed_dict)
                #     domain_acc, domain_acc_summary = session.run([accuracy_domain, accuracy_domain_summary])
                #     summary_writer_source_target.add_summary(domain_acc_summary, step)
                #
                #     # dssm discriminator
                #     feed_dict[name_training_domain] = 'source'
                #     domain_acc_dssm_op = session.run(accuracy_dssm_domain_op, feed_dict)
                #     domain_acc_dssm, domain_acc_dssm_summary = session.run([accuracy_dssm_domain, accuracy_dssm_domain_summary])
                #     summary_writer_source_target.add_summary(domain_acc_dssm_summary, step)
                #
                #     feed_dict[name_training_domain] = 'target'
                #     domain_acc_dssm_op = session.run(accuracy_dssm_domain_op, feed_dict)
                #     domain_acc_dssm, domain_acc_dssm_summary = session.run([accuracy_dssm_domain, accuracy_dssm_domain_summary])
                #     summary_writer_target_target.add_summary(domain_acc_dssm_summary, step)
                #
                #     feed_dict[name_training_domain] = 'domain_classifier_LR'
                #     domain_acc_dssm_op = session.run(accuracy_dssm_domain_op, feed_dict)
                #     domain_acc_dssm, domain_acc_dssm_summary = session.run([accuracy_dssm_domain, accuracy_dssm_domain_summary])
                #     summary_writer_source_source.add_summary(domain_acc_dssm_summary, step)
                # else:
                #     # Training discriminator for LR and DSSM
                #     feed_dict[isTrain] = True
                #     feed_dict[isTargetProduct] = False
                #     feed_dict[name_training_domain] = 'domain_classifier_LR'
                #     session.run([adam_train_step, adam_train_step_class_dssm], feed_dict)
                #     feed_dict[name_training_domain] = 'target'
                #     session.run([adam_train_step_class_dssm],feed_dict)
                #     feed_dict[name_training_domain] = 'source'
                #     session.run([adam_train_step_class_dssm],feed_dict)




p_fptr.close()
n_fptr.close()
#fp_val.close()
#fn_val.close()

