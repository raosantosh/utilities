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




QUERY_INDEX = 3
PRODCUTNAME_INDEX = 4
#PRODUCTDESC_INDEX = 2
PRODUCTBRAND_INDEX = 5




droupout_rate = 0.4

product_filters = [4, 4, 3]
query_filters = [3, 4]
num_filters_out = 256
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
vocabulary_char_size = 76



def getIntForChar(c):
    if c in char2Int:
        return char2Int[c]
    return vocabulary_char_size-1

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




def get_next_batch(positives_count):
    #
    # query_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    # productname_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    # brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))

    query_char_data = np.zeros((batch_size, query_max_length, vocabulary_char_size + 1, 1))
    product_char_data = np.zeros((batch_size, productname_max_length, vocabulary_char_size + 1, 1))

    product_index = 0
    end_of_file = False
    positive_lines =[]
    for index in range(batch_positives):
        positives_line = sys.stdin.readline()
        if not positives_line:
            end_of_file = True
            return query_char_data, product_char_data, \
           positive_lines, end_of_file, index

        positive_lines.append(positives_line)

        ptokens = positives_line.rstrip().split(separator)


        # if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX]) == 0):
        #     continue;
        #
        #
        # querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        # productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX], productname_max_length)
        # productbrandnormalized = ""
        #
        # if len(ptokens[PRODUCTBRAND_INDEX]) != 0:
        #     productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX], brand_max_length)
        #
        #
        #
        # features_query_triplets = getinputtriplets(querynormalized, 3,nb_triplets_query_product_buckets)
        # features_productname_triplets = getinputtriplets(productnamenormalized, 3,nb_triplets_query_product_buckets)
        # features_productbrand_triplets = getinputtriplets(productbrandnormalized, 3,nb_triplets_query_product_buckets)
        #
        #
        #
        #
        # for cidx in features_query_triplets:
        #     query_triplets[product_index, cidx, 0] = 1
        # for cidx in features_productname_triplets:
        #     productname_triplets[product_index, cidx, 0] = 1
        # for cidx in features_productbrand_triplets:
        #     brand_triplets[product_index, cidx, 0] = 1

        # deepmatch

        #querynormalized = normalizeQuery(ptokens[QUERY_INDEX], query_max_length)
        #productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX], productname_max_length)

        query_chars = query2tensor(ptokens[QUERY_INDEX])
        product_chars = product2tensor(ptokens[PRODCUTNAME_INDEX])
        for cidx in range(len(query_chars)):
            query_char_data[product_index, cidx, int(query_chars[cidx]), 0] = 1
        for cidx in range(len(product_chars)):
            product_char_data[product_index, cidx, int(product_chars[cidx]), 0] = 1

        product_index += 1

    return query_char_data, product_char_data, positive_lines, end_of_file, index



    return h, W



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




init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
#saver = tf.train.Saver(max_to_keep=5)
#summary_op = tf.summary.merge_all()


positives_count_test = 100

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:

    modelname = sys.argv[1]
    saver = tf.train.import_meta_graph(modelname+".meta")
    saver.restore(session, modelname)

    with tf.device('/cpu:0'):
        graph = tf.get_default_graph()
        session.run(init)
        session.run(init_local)
        end_of_file = False
        isTrain = graph.get_tensor_by_name("isTrain:0")
        query_input = graph.get_tensor_by_name("query:0")
        product_input = graph.get_tensor_by_name("product:0")
        #y_prediction = graph.get_tensor_by_name("pCTR_dssm:0")
        isTargetProduct = graph.get_tensor_by_name("isTargetProduct:0")
        name_training_domain = graph.get_tensor_by_name("domain:0")

        pCTR_deep_match = graph.get_tensor_by_name("pCTR_deepMatch:0")


        end_of_file = False




        while not end_of_file:
            query_char_batch_data, product_char_batch_data, \
            positive_lines, end_of_file, nb_positives = get_next_batch(positives_count_test)
            fd = {
                query_input: query_char_batch_data,
                product_input: product_char_batch_data,
                isTrain: False,
                name_training_domain: 'source',
                isTargetProduct: False
                }

            pctr = session.run(pCTR_deep_match, feed_dict=fd)
            for index in range(nb_positives):
                print(positive_lines[index].rstrip() + separator + str(pctr[index][0]))
    session.close()

