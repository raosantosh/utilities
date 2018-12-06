from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import mmh3
import time
from sets import Set

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
access_mode = "r"

#
# paths to filess
cwd = os.getcwd()
path_data = '/var/opt/amin/Data/datasets/'
path_model = cwd+'/../resources/datasets/models/'
positives_training_file = path_data+"positive_training_samples_query_productname_stemmed_131.csv"
negatives_training_file = path_data+"negative_training_samples_query_productname_stemmed_131.csv"
test_positives_file = path_data+"test_positives_unique_file_131"
test_negatives_file = path_data+"test_negatives_unique_file_131"    

annotation_file = cwd+'/../resources/datasets/annotations_fully_stemmed_all.csv'


max_positive_training_samples_size = 11381200
max_negative_training_samples_size = 27497500
nb_test_batches = 10

p_fptr = open(positives_training_file, mode=access_mode)
n_fptr = open(negatives_training_file, mode=access_mode)

fp = open(test_positives_file, access_mode)
fn = open(test_negatives_file, access_mode)

batch_positives = 32

negative_samples_factor = 6

positive_index = 0
negative_index = 0

query_max_length = 25
productname_max_length = 90
productdescription_max_length = 500
brand_max_length = 25


nb_triplets_query_product_buckets = 32768
first_layer_size = 300 #DSSM paper
second_layer_size = 100 #DSSM paper

batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 100

QUERY_INDEX = 0
PRODCUTNAME_INDEX_POSITIVE = 1
PRODCUTNAME_INDEX_NEGATIVE = 0

PRODUCTDESC_INDEX_POSITIVE = 2
PRODUCTDESC_INDEX_NEGATIVE = 1

PRODUCTBRAND_INDEX_POSITIVE = 3
PRODUCTBRAND_INDEX_NEGATIVE = 2
PRODUCTINMEMO = 5
PRODUCTANNOTATION = 6

droupout_rate = 0.4



def relativeImprovements(result, baseline):
    return (result-baseline)/baseline;

def printMetric(tp, fp, tn, fn):
    eps=0.0000001
    print('TP '+str(tp))
    print('FP '+str(fp))
    print('TN '+str(tn))
    print('FN '+str(fn))
    print('PRECISION '+str(float(tp)/float(tp+fp+eps)))
    print('RECALL '+str(float(tp)/float(tp+fn+eps)))
    print('Specificity  '+str(float(tn)/float(tn+fp+eps)))
    return float(tp)/float(tp+fp+eps), float(tp)/float(tp+fn+eps), float(tn)/float(tn+fp+eps)

def printForTS(predictions, TS):
    print('====================')
    print('Score for TS '+str(TS))
    tp = predictions[predictions.prediction>=TS][predictions.annotation>0].annotation.count()
    fn = predictions[predictions.prediction<TS][predictions.annotation>0].annotation.count()
    tn = predictions[predictions.prediction<TS][predictions.annotation<=0].annotation.count()
    fp = predictions[predictions.prediction>=TS][predictions.annotation<=0].annotation.count()
    precision_letters, recall_letter, sensitivity_letters = printMetric(tp, fp, tn, fn)

    print('====================')
    print('====================')

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


def get_next_batch():

    labels = np.zeros((batch_size, 1))

    query_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    productname_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))


    product_index = 0

    for index in range(batch_positives):
        positives_line = read_positives_file()
        ptokens = positives_line.rstrip().split('|')

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;


        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productbrandnormalized = ""

        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)



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
        negatives = 0

        while (negatives != negative_samples_factor):
            negatives_line = read_negatives_file()
            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;

            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            productbrandnormalized = ""


            if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) != 0:
                productbrandnormalized = normalizeProduct(ntokens[PRODUCTBRAND_INDEX_NEGATIVE], brand_max_length)

            features_query_triplets = getinputtriplets(querynormalized, 3, nb_triplets_query_product_buckets)
            features_productname_triplets = getinputtriplets(productnamenormalized, 3,
                                                             nb_triplets_query_product_buckets)
            features_productbrand_triplets = getinputtriplets(productbrandnormalized, 3,
                                                              nb_triplets_query_product_buckets)


            for cidx in features_query_triplets:
                query_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_triplets:
                productname_triplets[product_index, cidx, 0] = 1
            for cidx in features_productbrand_triplets:
                brand_triplets[product_index, cidx, 0] = 1


            product_index += 1
            negatives += 1

    for index in range(batch_size):
        if index % (negative_samples_factor + 1) == 0:
            labels[index, 0] = 1  # labels[index] = 1

    return labels, query_triplets, productname_triplets, brand_triplets


def get_next_test_data(pos_test_file_name, neg_test_file_name, positives_count):
    global fn
    global fp
    test_negative_samples_factor = negative_samples_factor
    test_data_size = positives_count * (test_negative_samples_factor + 1)
    labels = np.zeros((test_data_size, 1))

    query_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))


    product_index = 0
    if (~fp.closed):
        fp.close()
    lines = []
    fp = open(pos_test_file_name, 'r')
    for index in range(batch_positives):

        positives_line = fp.readline()
        if not positives_line:
            fp.close()
            fp = open(pos_test_file_name, 'r')
            positives_line = fp.readline()

        lines.append(positives_line)
        ptokens = positives_line.rstrip().split('|')

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;


        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productbrandnormalized = ""

        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)



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
        negatives = 0

        while (negatives != negative_samples_factor):
            negatives_line = read_negatives_file()
            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;

            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            productbrandnormalized = ""

            if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) != 0:
                productbrandnormalized = normalizeProduct(ntokens[PRODUCTBRAND_INDEX_NEGATIVE], brand_max_length)

            features_query_triplets = getinputtriplets(querynormalized, 3, nb_triplets_query_product_buckets)
            features_productname_triplets = getinputtriplets(productnamenormalized, 3,
                                                             nb_triplets_query_product_buckets)
            features_productbrand_triplets = getinputtriplets(productbrandnormalized, 3,
                                                              nb_triplets_query_product_buckets)


            for cidx in features_query_triplets:
                query_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_triplets:
                productname_triplets[product_index, cidx, 0] = 1
            for cidx in features_productbrand_triplets:
                brand_triplets[product_index, cidx, 0] = 1


            product_index += 1
            negatives += 1

    for index in range(batch_size):
        if index % (negative_samples_factor + 1) == 0:
            labels[index, 0] = 1  # labels[index] = 1

    return labels, query_triplets, productname_triplets, brand_triplets, lines

def get_next_annotation_data(annotation_file_name):
    global fn
    global fp

    test_data_size = 4768
    #labels = np.zeros((test_data_size, 1))
    annotations =  np.zeros((test_data_size, 1))
    in_memo =  np.zeros((test_data_size, 1))

    query_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))



    product_index = 0
    if (~fp.closed):
        fp.close()
    fp = open(annotation_file_name, 'r')
    for index in range(test_data_size):

        positives_line = fp.readline()
        if not positives_line:
            fp.close()
            fp = open(annotation_file_name, 'r')
            positives_line = fp.readline()

        ptokens = positives_line.rstrip().split('\t')

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;


        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productbrandnormalized = ""

        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)



        features_query_triplets = getinputtriplets(querynormalized, 3,nb_triplets_query_product_buckets)
        features_productname_triplets = getinputtriplets(productnamenormalized, 3,nb_triplets_query_product_buckets)
        features_productbrand_triplets = getinputtriplets(productbrandnormalized, 3,nb_triplets_query_product_buckets)

        for cidx in features_query_triplets:
            query_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_triplets:
            productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            brand_triplets[product_index, cidx, 0] = 1
        annotations[product_index,0]=int(ptokens[PRODUCTANNOTATION])
        in_memo[product_index,0]=int(ptokens[PRODUCTANNOTATION])

        product_index += 1



    return annotations, in_memo, query_triplets, productname_triplets, brand_triplets



def phi(x, n_output, droupout_rate, isTraining, name=None, activation=None, reuse=None,
        dropout=None):


    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(
            name='W'+name,
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b'+name,
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h'+name,
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        if dropout:
            h = tf.cond(isTraining, lambda: tf.layers.dropout(h, rate=droupout_rate, training=True),
                        lambda: tf.layers.dropout(h, rate=0.0, training=True))

    return h, W


def read_positives_file():
    global positive_index
    global p_fptr

    if positive_index == (max_positive_training_samples_size - 1):

        if (~p_fptr.closed):
            p_fptr.close()
            p_fptr = open(positives_training_file, mode=access_mode)

        positive_index = 0

    p_line = p_fptr.readline()
    positive_index += 1

    return p_line


def read_negatives_file():
    global negative_index
    global n_fptr

    if negative_index == (max_negative_training_samples_size - 1):

        if (~n_fptr.closed):
            n_fptr.close()
            n_fptr = open(negatives_training_file, access_mode)

        negative_index = 0

    n_line = n_fptr.readline()
    negative_index += 1

    return n_line


with tf.device('/cpu:0'):
    max_iterations = 20
    num_batches = max_positive_training_samples_size // (batch_size // (negative_samples_factor + 1))
    isTrain = tf.placeholder(tf.bool, shape=(), name='isTrain')


    query_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="query_triplets")
    query_triplets_emb_flat  = tf.reshape(query_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                               name="query_triplets_flat")


    query_out_1, _ = phi(query_triplets_emb_flat, n_output=first_layer_size,
                                                      droupout_rate=droupout_rate,
                                                      activation=tf.nn.tanh, name='query_triplets_emb_projection',
                                                      isTraining=isTrain,
                                                      dropout=True)

    query_out_2, _ = phi(query_out_1, n_output=second_layer_size,
                                                      droupout_rate=droupout_rate,
                                                      activation=tf.nn.tanh, name='query_2_triplets_emb_projection',
                                                      isTraining=isTrain,
                                                      dropout=True)


    productname_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                              name="productname_triplets")
    productname_triplets_emb_flat = tf.reshape(productname_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                               name="productname_triplets_flat")

    productbrand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                               name="productbrand_triplets")
    productbrand_triplets_emb_flat = tf.reshape(productbrand_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                name="productbrand_triplets_flat")

    product = tf.concat([productname_triplets_emb_flat,productbrand_triplets_emb_flat],1)


    product_out_1, _ = phi(product, n_output=first_layer_size,
                                                      droupout_rate=droupout_rate,
                                                      activation=tf.nn.tanh, name='product_triplets_emb_projection',
                                                      isTraining=isTrain,
                                                      dropout=True)

    product_out_2, _ = phi(product_out_1, n_output=second_layer_size,
                                                      droupout_rate=droupout_rate,
                                                      activation=tf.nn.tanh, name='product_2_triplets_emb_projection',
                                                      isTraining=isTrain,
                                                      dropout=True)




    y_true = tf.placeholder(tf.float32, shape=(None, 1))
    y_prediction = tf.reduce_sum(tf.multiply(query_out_2, product_out_2),1, keep_dims=True)

    pCTR = tf.nn.sigmoid(y_prediction, name='pCTR')

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction),name="cross_entropy")
    adam_train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

    # #metrics computation
    # with tf.name_scope('validation_metrics'):
    #     validation_cross_cum = tf.get_variable("validation_cross_entropy_cum", [batch_size, 1],dtype=tf.float32,initializer=tf.zeros_initializer)
    #
    #     validation_cross_cum =  tf.concat( [[cross_entropy_tensor], [validation_cross_cum]] , 1)[0]
    #     #cross_cum = tf.transpose(tf.concat([tf.transpose(cross_entropy_tensor), tf.transpose(cross_cum)], 1))
    #     validation_cross_entropy_metric = tf.reduce_mean(validation_cross_cum)
    #
    #     validation_auc, validation_auc_op = tf.metrics.auc(labels=y_true, predictions=pCTR)



init = tf.global_variables_initializer()
#init_local = tf.local_variables_initializer()
#metrics_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'validation_metrics']
#reset_metrics = [tf.initialize_variables(metrics_vars)]
saver = tf.train.Saver(max_to_keep=None)
start = time.time()


modelname = sys.argv[1]

with tf.Session() as session:
    session.run(init)
    #session.run(reset_metrics)

    #session.run(init_local)
    for iteration_index in range(max_iterations):

        for batch_index in range(1,num_batches+1):

            batch_labels, query_triplet_batch_data, productname_triplets_batch_data, \
            productbrand_triplets_batch_data = get_next_batch()



            if batch_index % 1000 == 0:
                auc_val=0
                avgloss=0
                auc_val2=0

                test_annotations, test_in_memo, test_query_triplets, test_productname_triplets, \
                test_productbraand_triplets_batch_data = get_next_annotation_data(annotation_file)

                fd_test = {query_triplets_emb: test_query_triplets,
                           productname_triplets_emb: test_productname_triplets,
                           productbrand_triplets_emb: test_productbraand_triplets_batch_data,
                           isTrain: False}


                y_score = session.run(
                     pCTR,
                    feed_dict=fd_test)
                df = pd.DataFrame(np.concatenate((test_annotations, y_score), axis=1))
                df.columns=[['annotation','prediction']]
                #print(df)

                for TS in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    printForTS(df,TS)
                #auc_val2 += roc_auc_score(test_labels, y_score)
                #avgloss+=loss

<<<<<<< HEAD
                avgloss = avgloss 
                auc_val2 = auc_val2

=======
                avgloss = avgloss
                auc_val2 = auc_val2
>>>>>>> d3833d33ddd46e558e8cf6bd3aa251cef4814ad6

                end = time.time()
                print('iteration ' + str(iteration_index) + ' auc@test  ' + str(auc_val2) +  ' loss@test '+ str(avgloss)  +' done ' + str(end - start))
                start = time.time()
                saver.save(session, path_model+'/'+modelname,global_step=batch_index * (iteration_index + 1))



            if batch_index % 100 == 0:
                end = time.time()
                training_loss = session.run(cross_entropy, feed_dict={y_true: batch_labels,
                                                                      query_triplets_emb: query_triplet_batch_data,
                                                                      productname_triplets_emb: productname_triplets_batch_data,
                                                                      productbrand_triplets_emb: productbrand_triplets_batch_data,
                                                                      isTrain: False})

                print('iteration ' + str(iteration_index) + ' batch ' + str(batch_index + 1) + ' loss ' + str(
                    training_loss) + ' done ' + str(end - start))

                start = time.time()
            # else:
            session.run(adam_train_step, feed_dict={y_true: batch_labels,
                                                                  query_triplets_emb: query_triplet_batch_data,
                                                                  productname_triplets_emb: productname_triplets_batch_data,
                                                                  productbrand_triplets_emb: productbrand_triplets_batch_data,
                                                                  isTrain: True})

if (~p_fptr.closed):
    p_fptr.close()

if (~n_fptr.closed):
    n_fptr.close()

if (~fp.closed):
    fp.close()

if (~fn.closed):
    fn.close()
