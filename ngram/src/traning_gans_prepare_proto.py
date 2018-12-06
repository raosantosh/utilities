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
positives_training_file = path_data + "positive_training_samples_query_productname_stemmed_101.csv"
negatives_training_file = path_data + "negative_training_samples_query_productname_stemmed_101.csv"
positives_validation_file = path_data + "positive_training_samples_query_productname_stemmed_101.csv"
negatives_validation_file = path_data + "negative_training_samples_query_productname_stemmed_101.csv"

p_fptr = CircularFile(positives_training_file)
n_fptr = CircularFile(negatives_training_file)
fp_val = CircularFile(positives_validation_file)
fn_val = CircularFile(negatives_validation_file)

max_positive_training_samples_size = 1000000

outputproto = path_data + "positive_negative_101.proto"

# Parameteres

learning_rate = 0.00005
learning_rate_gans = 0.00005
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
    test_labels = np.zeros(shape=(test_data_size),dtype=np.dtype(np.int8))

    #query_productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))


    query_triplets = np.zeros(shape=(test_data_size, nb_triplets_query_product_buckets),
                                            dtype=np.dtype(np.int8))

    productname_triplets = np.zeros(shape=(test_data_size, nb_triplets_query_product_buckets),
                                                  dtype=np.dtype(np.int8))

    # query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    #
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
        # features_query_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized,
        #                                                                          nb_triplets_query_product_buckets)
        for cidx in features_query_triplets:
            query_triplets[product_index, cidx] = 1
        for cidx in features_productname_triplets:
            productname_triplets[product_index, cidx] = 1
        # for cidx in features_query_productname_triplets:
        #     query_productname_triplets[product_index, cidx, 0] = 1
        product_index += 1
        negatives = 0

        while (negatives != test_negative_samples_factor):
            negatives_line = fnn.readline()
            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;
            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            #features_query_triplets = getinputtriplets(querynormalized, 3, nb_triplets_query_product_buckets)
            features_productname_triplets = getinputtriplets(productnamenormalized, 3,
                                                             nb_triplets_query_product_buckets)
            #features_query_productname_triplets = query2producttripletrepresentation(querynormalized,
            #                                                                         productnamenormalized,
            #                                                                         nb_triplets_query_product_buckets)
            for cidx in features_query_triplets:
                query_triplets[product_index, cidx] = 1
            for cidx in features_productname_triplets:
                productname_triplets[product_index, cidx] = 1
            # for cidx in features_query_productname_triplets:
            #     query_productname_triplets[product_index, cidx, 0] = 1
            product_index += 1
            negatives += 1
    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index] = 1
    return test_labels, query_triplets, productname_triplets






writer = tf.python_io.TFRecordWriter(outputproto)
num_batches = max_positive_training_samples_size // batch_size
print(' num total of batches is ' + str(num_batches + 1))

for batch_index in range(num_batches):
    batch_labels, query_triplets_batch_data, \
    productname_triplets_batch_data = get_next_test_data(batch_positives, negative_samples_factor,
                                                         p_fptr, n_fptr)
    batch = tf.train.Example(features=tf.train.Features(
        feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=batch_labels.flatten())),
                 'query': tf.train.Feature(int64_list=tf.train.Int64List(value=query_triplets_batch_data.flatten().nonzero()[0])),
                 'product': tf.train.Feature(int64_list=tf.train.Int64List(value=productname_triplets_batch_data.flatten().nonzero()[0]))
                 }))
    serialized_batch = batch.SerializeToString()
    writer.write(serialized_batch)
    if batch_index % 100 == 0:
        print(' batch ' + str(batch_index + 1))


def print_tfrecords(input_filename):
    for serialized_example in tf.python_io.tf_record_iterator(input_filename):
        # Get serialized example from file
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        label = np.array(example.features.feature["label"].int64_list.value)
        label = label.reshape(batch_size)
        query_index = np.array(example.features.feature["query"].int64_list.value)
        query = np.zeros((batch_size * nb_triplets_query_product_buckets))
        query[query_index] = 1
        query = query.reshape(batch_size, nb_triplets_query_product_buckets)
        print(query)




p_fptr.close
n_fptr.close
fp_val.close
fn_val.close

