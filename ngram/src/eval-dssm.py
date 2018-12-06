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



separator = '|'
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


nb_triplets_query_product_buckets = 32768


batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 10
droupout_rate = 0.0




QUERY_INDEX = 0
PRODCUTNAME_INDEX = 1
#PRODUCTDESC_INDEX = 2
PRODUCTBRAND_INDEX = 3




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
    tokens=query.rstrip().split(' ')
    for token in tokens:
        token="#"+token+"#"
        for i in range(len(token) - length +1):
            triplets.add(token[i:i + length])
            #triplets.add(token[i:i + length+1])
    return triplets

def getinputtriplets(input, len, BUCKETS):
    features = Set()
    for triplet in getTriplets(input, len):
        features.add(abs(int(mmh3.hash(triplet))) % BUCKETS)
    return features

def query2producttripletrepresentation(query, product, len, BUCKETS):
    features = Set()
    qgrams_4 = getTriplets(query, len)
    pgrams_4 = getTriplets(product, len)
    for gram in qgrams_4:
        if gram in pgrams_4:
            features.add(abs(int(mmh3.hash(gram))) % BUCKETS)
    return features


def get_next_batch(positives_count):

    query_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    productname_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))


    product_index = 0
    end_of_file = False
    positive_lines =[]
    for index in range(batch_positives):
        positives_line = sys.stdin.readline()
        if not positives_line:
            end_of_file = True
            return query_triplets, productname_triplets, brand_triplets, \
           positive_lines, end_of_file, index

        positive_lines.append(positives_line)

        ptokens = positives_line.rstrip().split(separator)


        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX]) == 0):
            continue;


        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX], productname_max_length)
        productbrandnormalized = ""

        if len(ptokens[PRODUCTBRAND_INDEX]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX], brand_max_length)



        features_query_triplets = getinputtriplets(querynormalized, 3,nb_triplets_query_product_buckets)
        features_productname_triplets = getinputtriplets(productnamenormalized, 3,nb_triplets_query_product_buckets)
        features_productbrand_triplets = getinputtriplets(productbrandnormalized, 3,nb_triplets_query_product_buckets)




        for cidx in features_query_triplets:
            query_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_triplets:
            productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            brand_triplets[product_index, cidx, 0] = 1

        product_index += 1

    return query_triplets, productname_triplets, brand_triplets, positive_lines, end_of_file, index











init = tf.global_variables_initializer()



positives_count_test = 100

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
    modelname = sys.argv[1]
    saver = tf.train.import_meta_graph(modelname+".meta")
    saver.restore(session, modelname)

    with tf.device('/cpu:0'):
        graph = tf.get_default_graph()
        end_of_file = False
        isTrain = graph.get_tensor_by_name("isTrain:0")
        query_triplets_emb = graph.get_tensor_by_name("query_triplets:0")
        productname_triplets_emb = graph.get_tensor_by_name("productname_triplets_dssm:0")
        #productbrand_triplets_emb = graph.get_tensor_by_name("productbrand_triplets:0")
        y_prediction = graph.get_tensor_by_name("pCTR:0")
        isTargetProduct = graph.get_tensor_by_name("isTargetProduct:0")
        name_training_domain = graph.get_tensor_by_name("domain:0")



        while not end_of_file:
            query_triplet_batch_data, productname_triplets_batch_data, \
            productbrand_triplets_batch_data, positive_lines, end_of_file, nb_positives = get_next_batch(positives_count_test)
            fd = {
                         query_triplets_emb: query_triplet_batch_data,
                         productname_triplets_emb: productname_triplets_batch_data,
                         #productbrand_triplets_emb: productbrand_triplets_batch_data,
                         isTrain: False,
                         name_training_domain: 'source', isTargetProduct: False
                }

            pctr = session.run(y_prediction, feed_dict=fd)
            for index in range(nb_positives):
                print(positive_lines[index].rstrip() + separator + str(pctr[index][0]))
    session.close()

