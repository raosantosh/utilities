from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os
import mmh3
from sets import Set

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if len(sys.argv) < 2:
    print('please give model directory as parameter')
    sys.exit(-1)



separator = '\t'
max_positive_training_samples_size = 10000
max_negative_training_samples_size = 31503552


batch_positives = 8

negative_samples_factor = 31

positive_index = 0
negative_index = 0


query_max_length = 25
product_max_length = 90

nb_triplets_query_product_buckets = 32768


batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 10
droupout_rate = 0.0




QUERY_INDEX = 1
PRODCUT_INDEX = 2

#QUERY_INDEX = 0
#PRODCUT_INDEX = 1

droupout_rate = 0.4

def normalizeQuery(query, maxlength):

    query=query.replace("+"," ")
    query=query.replace("|","")
    query=query.strip()
    query=query.lower()
    if len(query)>maxlength:
        query=query[0:maxlength]
    query=query.strip()
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
    if len(product)>maxlength:

        product=product[0:maxlength]

    product = product.lower()
    product=product.strip()
    return product






def getTriplets(query, length):
    triplets = Set()
    for i in range(len(query) - length +1):
        triplets.add(query[i:i + length])
    return triplets


def query2producttripletrepresentation(query, product, BUCKETS):
    features = Set()
    qgrams_4 = getTriplets(query, 4)
    qgrams_3 = getTriplets(query, 3)
    qgrams_4 = qgrams_4.union(qgrams_3)


    pgrams_4 = getTriplets(product, 3)
    pgrams_3 = getTriplets(product, 4)
    pgrams_4 = pgrams_4.union(pgrams_3)

    pq_grams = pgrams_4.intersection(qgrams_4)
    for gram in pq_grams:
        features.add(abs(int(mmh3.hash(gram))) % BUCKETS)
    return features




def phi(x, n_output, droupout_rate, isTraining, name=None, activation=None, reuse=None, batch_normalization=None,
        dropout=None):
    # if len(x.get_shape()) != 2:
    #     x = flatten(x, reuse=reuse)

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


        if activation:
            h = activation(h)

        if dropout:
            h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                        lambda: tf.layers.dropout(h, rate=0.0, training=True))

    return h, W


def get_next_batch(positives_count):
    test_data_size = positives_count

    query_product_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    query_index = 0
    product_index = 0

    positive_lines = []

    end_of_file = False
    for index in range(positives_count):
        positives_line = sys.stdin.readline()
        if not positives_line:
            end_of_file = True
            return query_product_triplets, positive_lines, end_of_file, index
        positive_lines.append(positives_line)

        ptokens = positives_line.rstrip().split(separator)
        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUT_INDEX]) == 0):
            continue;


        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnormalized = normalizeQuery(ptokens[PRODCUT_INDEX],product_max_length)
        features_triplets_qp = query2producttripletrepresentation(querynormalized, productnormalized, nb_triplets_query_product_buckets)
        for cidx in features_triplets_qp:
            query_product_triplets[product_index, cidx, 0] = 1


        product_index += 1

    return query_product_triplets, positive_lines, end_of_file, index


with tf.device('/cpu:0'):
    max_iterations = 200

    num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))

    isTrain = tf.placeholder(tf.bool, shape=())

    query_product_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                name="query_product_triplets")
    query_product_triplets_emb_flat = tf.reshape(query_product_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                 name="query_product_triplets_flat")
    query_product = tf.concat([query_product_triplets_emb_flat], 1)
    query_product_out_2, query_product_out_wt_2 = phi(query_product, n_output=1, droupout_rate=droupout_rate,
                                                      activation=None, name='query_fc_layer_2',
                                                      isTraining=isTrain,
                                                      batch_normalization=False, dropout=True)

    y_prediction = tf.nn.sigmoid(query_product_out_2)



init = tf.global_variables_initializer()

saver = tf.train.Saver()

positives_count_test = 64

with tf.Session() as session:
    modelname = sys.argv[1]

    saver.restore(session, modelname)
    end_of_file = False
    while not end_of_file:
        query_product_triplets, positive_lines, end_of_file, nb_positives = get_next_batch(positives_count_test)
        fd = {query_product_triplets_emb: query_product_triplets, isTrain: False}
        pctr = session.run(y_prediction, feed_dict=fd)
        for index in range(nb_positives):
            print(positive_lines[index].rstrip() + separator + str(pctr[index][0]))
    session.close()


