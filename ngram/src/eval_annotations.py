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
productname_max_length = 90
productdescription_max_length = 500
brand_max_length = 25

nb_triplets_query_product_buckets = 2**15


batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 10
droupout_rate = 0.0



QUERY_INDEX = 0
PRODCUTNAME_INDEX = 1
PRODUCTBRAND_INDEX = 3







droupout_rate = 0.4

################
# This part of the code covers
# how features are extracted from query and product name, and brand
# to produce a vector embedding the query-product pair
# each 3-letters that is in both name and query will
# fire a positif bit in the index corresponding to this 3-letters
# the index is computed using a hash of the 3-letters
################


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
    tokens=query.rstrip().split(' ')
    for token in tokens:
        token="#"+token+"#"
        for i in range(len(token) - length +1):
            triplets.add(token[i:i + length])
    return triplets


def query2producttripletrepresentation(query, product, BUCKETS):
    features = Set()
    qgrams_4 = getTriplets(query, 3)
    pgrams_4 = getTriplets(product, 3)
    for gram in qgrams_4:
        if gram in pgrams_4:
            features.add(abs(int(mmh3.hash(gram))) % BUCKETS)
    return features




# def phi(x, n_output, droupout_rate, isTraining, name=None, activation=None, reuse=None, batch_normalization=None,
#         dropout=None):
#     # if len(x.get_shape()) != 2:
#     #     x = flatten(x, reuse=reuse)
#
#     n_input = x.get_shape().as_list()[1]
#
#     bn_epsilon = 1e-3
#     normal_axes = [0]
#
#     with tf.variable_scope(name, reuse=reuse):
#         W = tf.get_variable(
#             name='W',
#             shape=[n_input, n_output],
#             dtype=tf.float32,
#             initializer=tf.contrib.layers.xavier_initializer())
#
#         b = tf.get_variable(
#             name='b',
#             shape=[n_output],
#             dtype=tf.float32,
#             initializer=tf.constant_initializer(0.0))
#
#         h = tf.nn.bias_add(
#             name='h',
#             value=tf.matmul(x, W),
#             bias=b)
#
#
#         if activation:
#             h = activation(h)
#
#         if dropout:
#             h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
#                         lambda: tf.layers.dropout(h, rate=0.0, training=True))
#
#     return h, W


# def get_next_batch(positives_count):
#     test_data_size = positives_count
#
#     query_product_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
#
#     query_index = 0
#     product_index = 0
#
#     positive_lines = []
#
#     end_of_file = False
#     for index in range(positives_count):
#         positives_line = sys.stdin.readline()
#         if not positives_line:
#             end_of_file = True
#             return query_product_triplets, positive_lines, end_of_file, index
#         positive_lines.append(positives_line)
#
#         ptokens = positives_line.rstrip().split(separator)
#         if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUT_INDEX]) == 0):
#             continue;
#
#
#         querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
#         productnormalized = normalizeQuery(ptokens[PRODCUT_INDEX],product_max_length)
#         features_triplets_qp = query2producttripletrepresentation(querynormalized, productnormalized, nb_triplets_query_product_buckets)
#         for cidx in features_triplets_qp:
#             query_product_triplets[product_index, cidx, 0] = 1
#
#
#         product_index += 1
#
#     return query_product_triplets, positive_lines, end_of_file, index


# with tf.device('/cpu:0'):
#     max_iterations = 200
#
#     num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))
#
#     isTrain = tf.placeholder(tf.bool, shape=())
#
#     query_product_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
#                                                 name="query_product_triplets")
#     query_product_triplets_emb_flat = tf.reshape(query_product_triplets_emb, [-1, nb_triplets_query_product_buckets],
#                                                  name="query_product_triplets_flat")
#     query_product = tf.concat([query_product_triplets_emb_flat], 1)
#     query_product_out_2, query_product_out_wt_2 = phi(query_product, n_output=1, droupout_rate=droupout_rate,
#                                                       activation=None, name='query_fc_layer_2',
#                                                       isTraining=isTrain,
#                                                       batch_normalization=False, dropout=True)
#
#     y_prediction = tf.nn.sigmoid(query_product_out_2)


def get_next_batch(positives_count):


    test_data_size = positives_count
    #labels = np.zeros((batch_size, 1))
    query_productname_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    # query_productdescription_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))

    # query_productname_description_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    # query_productdescription_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    # query_productname_description_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))

    product_index = 0
    positive_lines = []

    end_of_file = False

    for index in range(batch_positives):
        positives_line = sys.stdin.readline()
        if not positives_line:
            end_of_file = True
            return query_productname_triplets,  query_brand_triplets,\
           query_productname_brand_triplets, \
           positive_lines, end_of_file, index

        positive_lines.append(positives_line)

        ptokens = positives_line.rstrip().split(separator)
        if len(ptokens)<PRODCUTNAME_INDEX+1:
            continue;
        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX], productname_max_length)
        # productdescriptionnormalized = []
        productbrandnormalized =""
        # if len(ptokens[PRODUCTDESC_INDEX_POSITIVE]) != 0:
        #     productdescriptionnormalized = normalizeProduct(ptokens[PRODUCTDESC_INDEX_POSITIVE], productdescription_max_length)

        if len(ptokens[PRODUCTBRAND_INDEX]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX], brand_max_length)

        features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, nb_triplets_query_product_buckets)
        # features_productdecription_triplets = query2producttripletrepresentation(querynormalized, productdescriptionnormalized, nb_triplets_query_product_buckets)
        features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized, nb_triplets_query_product_buckets)
        # features_productname_description_triplets  = features_productname_triplets.intersection(features_productdecription_triplets)
        features_productname_brand_triplets  = features_productname_triplets.intersection(features_productbrand_triplets)
        # features_productdescription_brand_triplets  = features_productdecription_triplets.intersection(features_productname_brand_triplets)
        # features_productname_description_brand_triplets = features_productname_brand_triplets.intersection(features_productdecription_triplets)

        for cidx in features_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            query_brand_triplets[product_index, cidx, 0] = 1
        # for cidx in features_productdecription_triplets:
        #     query_productdescription_triplets[product_index, cidx, 0] = 1

        # for cidx in features_productname_description_triplets:
        #     query_productname_description_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_brand_triplets:
            query_productname_brand_triplets[product_index, cidx, 0] = 1
        # for cidx in features_productdescription_brand_triplets:
        #     query_productdescription_brand_triplets[product_index, cidx, 0] = 1
        # for cidx in features_productname_description_brand_triplets:
        #     query_productname_description_brand_triplets[product_index, cidx, 0] = 1


        product_index += 1
        negatives = 0


    return query_productname_triplets, query_brand_triplets , query_productname_brand_triplets,\
           positive_lines, end_of_file, index





init = tf.global_variables_initializer()

#saver = tf.train.Saver()

positives_count_test = 64

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
    modelname = sys.argv[1]
    saver = tf.train.import_meta_graph(modelname+".meta")
    saver.restore(session, modelname)
    with tf.device('/cpu:0'):
        graph = tf.get_default_graph()
        end_of_file = False
        isTrain = graph.get_tensor_by_name("isTrain:0")
        query_productname_triplets_emb = graph.get_tensor_by_name("query_productname_triplets:0")
        query_productbrand_triplets_emb = graph.get_tensor_by_name("query_productbrand_triplets:0")
        query_productname_brand_triplets_emb = graph.get_tensor_by_name("query_productname_brand_triplets:0")
        y_prediction = graph.get_tensor_by_name("pCTR:0")

        while not end_of_file:
            query_productname_triplets_batch_data, query_productbrand_triplets_batch_data, \
            query_productname_brand_triplets_batch_data, positive_lines, end_of_file, nb_positives = get_next_batch(positives_count_test)

            fd = {
                         query_productname_triplets_emb: query_productname_triplets_batch_data,
                         query_productbrand_triplets_emb: query_productbrand_triplets_batch_data,
                         query_productname_brand_triplets_emb: query_productname_brand_triplets_batch_data,
                         isTrain: False}
            pctr = session.run(y_prediction, feed_dict=fd)
            for index in range(nb_positives):
                print(positive_lines[index].rstrip() + separator + str(pctr[index][0]))
    session.close()






