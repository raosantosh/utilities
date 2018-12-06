from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import mmh3
from sets import Set

import numpy as np
import tensorflow as tf
from myutils import CircularFile

access_mode = "r"

# Training - validation files location
#cwd = os.getcwd()
cwd = '/home/a.mantrach/LettersQuerySkuModel/resources/'
log_location = cwd + 'logs/'
path_data = cwd+'datasets/'
path_model = cwd + 'models/'
positives_training_file = path_data + "ngram_positive_training_samples_formatted.csv"
negatives_training_file = path_data + "ngram_negative_training_skus_formatted.csv"
outputproto = path_data + "ngram_training_skus_formatted_lem.proto"

with open(positives_training_file) as f:
    max_positive_training_samples_size=len(f.readlines())

p_fptr = CircularFile(positives_training_file)
n_fptr = CircularFile(negatives_training_file)


batch_positives = 4
negative_samples_factor = 7
query_max_length = 25
productname_max_length = 90
brand_max_length = 25
nb_triplets_query_product_buckets = 2**15
batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 100

QUERY_INDEX = 2
PRODCUTNAME_INDEX_POSITIVE = 3
PRODUCTBRAND_INDEX_POSITIVE = 4
PRODCUTNAME_INDEX_NEGATIVE = 1
PRODUCTBRAND_INDEX_NEGATIVE = 2



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

###
# we are using a mini-batch gradient descent based training.
# We read batches, and updates the parameters
# each batch is composed of positive query-product pair
# that are coming from the click log
# negative are constructed by associating a query to a random product
###
def next_batch(positives_count, test_negative_samples_factor, fpp, fnn):

    test_data_size = positives_count * (test_negative_samples_factor + 1)
    test_labels = np.zeros(shape=(test_data_size, 1),dtype=np.dtype(np.int8))

    query_productname_triplets = np.zeros(shape=(test_data_size, nb_triplets_query_product_buckets, 1),dtype=np.dtype(np.int8))
    query_brand_triplets = np.zeros(shape=(test_data_size, nb_triplets_query_product_buckets, 1),dtype=np.dtype(np.int8))
    query_productname_brand_triplets = np.zeros(shape=(test_data_size, nb_triplets_query_product_buckets, 1),dtype=np.dtype(np.int8))

    product_index = 0

    for index in range(positives_count):
        positives_line = fpp.readline()


        ptokens = positives_line.rstrip().split('|')

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;

        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productbrandnormalized =""
        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)

        features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, nb_triplets_query_product_buckets)
        features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized, nb_triplets_query_product_buckets)
        features_productname_brand_triplets  = features_productname_triplets.intersection(features_productbrand_triplets)

        for cidx in features_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            query_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_brand_triplets:
            query_productname_brand_triplets[product_index, cidx, 0] = 1

        product_index += 1
        negatives = 0

        while (negatives != test_negative_samples_factor):
            negatives_line = fnn.readline()
            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;

            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            productbrandnormalized = ""
            if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) != 0:
                productbrandnormalized = normalizeProduct(ntokens[PRODUCTBRAND_INDEX_NEGATIVE], brand_max_length)

            features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized,
                                                                                nb_triplets_query_product_buckets)
            features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized,
                                                                                nb_triplets_query_product_buckets)

            features_productname_brand_triplets = features_productname_triplets.intersection(
                features_productbrand_triplets)
            for cidx in features_productname_triplets:
                query_productname_triplets[product_index, cidx, 0] = 1
            for cidx in features_productbrand_triplets:
                query_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_brand_triplets:
                query_productname_brand_triplets[product_index, cidx, 0] = 1
            product_index += 1
            negatives += 1

    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index, 0] = 1
    return test_labels, query_productname_triplets,query_brand_triplets,query_productname_brand_triplets





writer = tf.python_io.TFRecordWriter(outputproto)
max_iterations = 5
num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))

print(' nb of batches ' + str(num_batches))
#for iteration_index in range(max_iterations):
for batch_index in range(num_batches):
    batch_labels, query_productname_triplets_batch_data, query_productbrand_triplets_batch_data, \
    query_productname_brand_triplets_batch_data = next_batch(batch_positives,
                                                             negative_samples_factor,
                                                             p_fptr, n_fptr)
    batch = tf.train.Example(features=tf.train.Features(
        feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=batch_labels.flatten())),
                 'query_productname': tf.train.Feature(int64_list=tf.train.Int64List(value=query_productname_triplets_batch_data.flatten().nonzero()[0])),
                 'query_productbrand': tf.train.Feature(int64_list=tf.train.Int64List(value=query_productbrand_triplets_batch_data.flatten().nonzero()[0])),
                 'query_productname_brand': tf.train.Feature(int64_list=tf.train.Int64List(value=query_productname_brand_triplets_batch_data.flatten().nonzero()[0]))
                 }))
    serialized_batch = batch.SerializeToString()
    writer.write(serialized_batch)
    if batch_index % 1000 == 0:
        print(' batch ' + str(batch_index + 1))

writer.close()

if (~p_fptr.closed):
    p_fptr.close

if (~n_fptr.closed):
    n_fptr.close


