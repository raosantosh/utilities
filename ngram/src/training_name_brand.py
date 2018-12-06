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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
access_mode = "r"

# Training - validation files location
#cwd = os.getcwd()
cwd = '/home/a.mantrach/LettersQuerySkuModel/resources/'
log_location = cwd + 'logs/'
path_data = cwd+'datasets/'
path_model = cwd + 'models/'
positives_training_file = path_data + "ngram_positive_training_samples_formatted.csv"
negatives_training_file = path_data + "ngram_negative_training_skus_formatted.csv"
annotation_file = cwd+'datasets/annotations_fully_stemmed_all.csv'

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

QUERY_INDEX = 2
PRODCUTNAME_INDEX_POSITIVE = 3
PRODUCTBRAND_INDEX_POSITIVE = 4
PRODCUTNAME_INDEX_NEGATIVE = 1
PRODUCTBRAND_INDEX_NEGATIVE = 2




def metricsAtTS(y_score, y_true, TS):
    eps=0.00000001
    tp = len(y_true[y_score >= TS][y_true[y_score >= TS]==1])
    fp = len(y_true[y_score >= TS][y_true[y_score >= TS]==0])
    tn = len(y_true[y_score < TS][y_true[y_score < TS]==0])
    fn = len(y_true[y_score < TS][y_true[y_score < TS]==1])
    return float(tp) / float(tp + fp + eps), float(tp) / float(tp + fn + eps), float(tn) / float(tn + fp + eps)


def scoresForMemo(y_score, y_true):
    eps=0.00000001
    tp = len(y_true[y_score == 1][y_true[y_score == 1] == 1])
    fp = len(y_true[y_score == 1][y_true[y_score == 1] == 0])
    tn = len(y_true[y_score == 0][y_true[y_score == 0] == 0])
    fn = len(y_true[y_score == 0][y_true[y_score == 0] == 1])
    precision_memo, recall_memo, specificiy_memo = float(tp) / float(tp + fp + eps), \
                                                   float(tp) / float(tp + fn + eps), \
                                                   float(tn) / float(tn + fp + eps)
    return precision_memo, recall_memo, specificiy_memo

def computeBestTS(y_score, y_true, precision_memo, specificiy_memo):
    for TS in np.arange(0.0, 1.0, 0.001):
        precision, recall, specificity = metricsAtTS(y_score,
                                                    y_true, TS)
        if precision>precision_memo and specificity>specificiy_memo:
            return precision, recall, specificity, TS





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
    test_labels = np.zeros((test_data_size, 1))

    query_productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

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

#
def get_next_annotation_data(annotation_file_name):
    QUERY_INDEX_ANNOTATION=0
    PRODCUTNAME_INDEX_ANNOTATION=1
    PRODCUTBRAND_INDEX_ANNOTATION=3
    PRODUCTINMEMO = 5
    PRODUCTANNOTATION = 6
    test_data_size = 4768
    annotations =  np.zeros((test_data_size, 1))
    in_memo =  np.zeros((test_data_size, 1))
    query_productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    product_index = 0
    fp = open(annotation_file_name, 'r')
    for index in range(test_data_size):
        positives_line = fp.readline()
        if not positives_line:
            fp.close()
            fp = open(annotation_file_name, 'r')
            positives_line = fp.readline()
        ptokens = positives_line.rstrip().split('\t')
        if (len(ptokens[QUERY_INDEX_ANNOTATION]) == 0 or len(ptokens[PRODCUTNAME_INDEX_ANNOTATION]) == 0):
            continue;
        querynormalized = normalizeQuery(ptokens[QUERY_INDEX_ANNOTATION],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_ANNOTATION], productname_max_length)
        productbrandnormalized =""
        if len(ptokens[PRODCUTBRAND_INDEX_ANNOTATION]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODCUTBRAND_INDEX_ANNOTATION], brand_max_length)
        features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, nb_triplets_query_product_buckets)
        features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized, nb_triplets_query_product_buckets)
        features_productname_brand_triplets  = features_productname_triplets.intersection(features_productbrand_triplets)
        for cidx in features_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            query_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_brand_triplets:
            query_productname_brand_triplets[product_index, cidx, 0] = 1
        annotations[product_index,0]=int(ptokens[PRODUCTANNOTATION])
        in_memo[product_index,0]=int(ptokens[PRODUCTINMEMO])
        product_index += 1
    return annotations, in_memo, query_productname_triplets, query_brand_triplets, query_productname_brand_triplets


##
# h = w*x + b (linear projection)
##
def linear(x, droupout_rate, isTraining, name=None,  reuse=None,
        dropout=None):

    n_input = x.get_shape().as_list()[1]
    n_output = 1
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

        if dropout:
            h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                        lambda: tf.layers.dropout(h, rate=0.0, training=True))

    return h, W




with tf.device('/cpu:0'):
    max_iterations = 5
    num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))
    isTrain = tf.placeholder(tf.bool, shape=(), name="isTrain")
    query_productname_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                    name="query_productname_triplets")
    query_productname_triplets_emb_flat = tf.reshape(query_productname_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                     name="query_productname_triplets_flat")
    query_productbrand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                    name="query_productbrand_triplets")
    query_productbrand_triplets_emb_flat = tf.reshape(query_productbrand_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                     name="query_productbrand_triplets_flat")
    query_productname_brand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="query_productname_brand_triplets")
    query_productname_brand_triplets_emb_flat = tf.reshape(query_productname_brand_triplets_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="query_productname_brand_triplets_flat")
    query_product = tf.concat([query_productname_triplets_emb_flat,
                               query_productbrand_triplets_emb_flat,
                               query_productname_brand_triplets_emb_flat], 1)
    query_product_out_2, query_product_out_wt_2 = linear(query_product, droupout_rate=droupout_rate,
                                                      name='query_fc_layer_2',
                                                      isTraining=isTrain,
                                                      dropout=True)

    y_true = tf.placeholder(tf.float32, shape=(None, 1))
    y_prediction = query_product_out_2
    pCTR = tf.nn.sigmoid(y_prediction, name="pCTR")
    auc_tf, auc_op = tf.metrics.auc(y_true,pCTR,name="rocAUC")
    auc_summary  = tf.summary.scalar("AUC on Training", auc_tf)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction))
    cross_entropy_summary = tf.summary.scalar("Cross Entropy on Training", cross_entropy)
    adam_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)



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
    summary_writer_source_target = tf.summary.FileWriter(log_location,session.graph)
    step = 0

    validation_annotations, validation_in_memo, validation_query_productname_triplets, \
    validation_query_productbrand_triplets_batch_data, \
    validation_query_productname_brand_triplets_batch_data = get_next_annotation_data(annotation_file)
    validation_annotations[validation_annotations > 0] = 1
    validation_annotations[validation_annotations <= 0] = 0
    fd_validation = {query_productname_triplets_emb: validation_query_productname_triplets,
               query_productbrand_triplets_emb: validation_query_productbrand_triplets_batch_data,
               query_productname_brand_triplets_emb: validation_query_productname_brand_triplets_batch_data,
               isTrain: False}
    precision_memo, recall_memo, specificiy_memo = scoresForMemo(validation_in_memo, validation_annotations)

    for iteration_index in range(max_iterations):
        for batch_index in range(num_batches):
            step += 1

            if batch_index % 100 == 0:
                test_labels, test_query_productname_triplets, test_query_productbrand_triplets_batch_data , \
                test_query_productname_brand_triplets_batch_data = next_batch(batch_positives*5,31,
                                                                                            p_fptr, n_fptr)

                fd_test = {y_true: test_labels, query_productname_triplets_emb: test_query_productname_triplets,
                           query_productbrand_triplets_emb:  test_query_productbrand_triplets_batch_data,
                           query_productname_brand_triplets_emb: test_query_productname_brand_triplets_batch_data,
                           isTrain: False}

                validation_loss, val_loss_summary, _ = session.run(
                    [cross_entropy, cross_entropy_summary, auc_op],
                    feed_dict=fd_test)
                summary_writer_source_target.add_summary(val_loss_summary   ,step)
                auc_roc, auc_roc_summary  = session.run([auc_tf, auc_summary])
                summary_writer_source_target.add_summary(auc_roc_summary,step)


                y_score = session.run(pCTR,feed_dict=fd_validation)
                auc=roc_auc_score(validation_annotations, y_score)

                summary = tf.Summary()
                summary.value.add(tag='AUC on Annotations', simple_value=auc)
                summary_writer_source_target.add_summary(summary, step)
                map=average_precision_score(validation_annotations, y_score)
                summary = tf.Summary()
                summary.value.add(tag='MAP on Annotations', simple_value=map)
                summary_writer_source_target.add_summary(summary, step)

                precision, recall, specificity, TS = computeBestTS(y_score, validation_annotations, precision_memo, specificiy_memo)

                summary = tf.Summary()
                summary.value.add(tag='Precision on Annotations', simple_value=precision)
                summary_writer_source_target.add_summary(summary, step)

                summary = tf.Summary()
                summary.value.add(tag='Recall on Annotations ', simple_value=recall)
                summary_writer_source_target.add_summary(summary, step)

                summary = tf.Summary()
                summary.value.add(tag='Specificity on Annotations', simple_value=specificity)
                summary_writer_source_target.add_summary(summary, step)
                summary_writer_source_target.flush()

                if batch_index % 1000 == 0:
                    saver.save(session, path_model+modelname,global_step=step)
            else:
                batch_labels, query_productname_triplets_batch_data, query_productbrand_triplets_batch_data, \
                query_productname_brand_triplets_batch_data = next_batch(batch_positives,
                                                                                 negative_samples_factor,
                                                                                 p_fptr, n_fptr)
                session.run(adam_train_step, feed_dict={y_true: batch_labels,
                                                                      query_productname_triplets_emb: query_productname_triplets_batch_data,
                                                                      query_productbrand_triplets_emb: query_productbrand_triplets_batch_data,
                                                                      query_productname_brand_triplets_emb: query_productname_brand_triplets_batch_data,
                                                                      isTrain: True})


p_fptr.close()
n_fptr.close()


