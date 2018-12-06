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
log_location = cwd + './logs/'
path_data = '/var/opt/amin/Data/'
queries_source_embeddings = path_data + "queries_164_embeddings.csv"
queries_target_embeddings = path_data + "queries_180_embeddings.csv"

product_source_positive_embeddings = path_data + "productnames_164_embeddings.csv"
product_target_positive_embeddings = path_data + "productnames_180_embeddings.csv"

product_source_negative_embeddings = path_data + "productnames_164_negative_embeddings.csv"
product_target_negative_embeddings = path_data + "productnames_180_negative_embeddings.csv"

with open(product_source_positive_embeddings) as f:
    max_positive_training_samples_size=len(f.readlines())

print('max size is '+str(max_positive_training_samples_size))
queries_source_embeddings_file =  CircularFile(queries_source_embeddings)
queries_target_embeddings_file =  CircularFile(queries_target_embeddings)
product_source_positive_embeddings_file =  CircularFile(product_source_positive_embeddings)
product_target_positive_embeddings_file =  CircularFile(product_target_positive_embeddings)
product_source_negative_embeddings_file =  CircularFile(product_source_negative_embeddings)
product_target_negative_embeddings_file =  CircularFile(product_target_negative_embeddings)

embeddings_size = 100

outputproto = path_data + "positive_negative_embeddings_164.proto"

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




writer = tf.python_io.TFRecordWriter(outputproto)
num_batches = max_positive_training_samples_size // batch_size
print(' num total of batches is ' + str(num_batches + 1))

for batch_index in range(num_batches):
    batch_labels, query_embeddings, \
    product_embeddings = get_next_test_data_embeddings(batch_positives, negative_samples_factor,
                                                       queries_source_embeddings_file,
                                                       product_source_positive_embeddings_file,
                                                       product_source_negative_embeddings_file)
    batch = tf.train.Example(features=tf.train.Features(
        feature={'label': tf.train.Feature(float_list=tf.train.FloatList(value=batch_labels.flatten())),
                 'query': tf.train.Feature(float_list=tf.train.FloatList(value=query_embeddings.flatten().nonzero()[0])),
                 'product': tf.train.Feature(float_list=tf.train.FloatList(value=product_embeddings.flatten().nonzero()[0]))
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




queries_source_embeddings_file.close

product_source_positive_embeddings_file.close

product_source_negative_embeddings_file.close