import tensorflow as tf
import numpy as np
import sys
#import pandas as pd
import mmh3
from sets import Set
import json


query_max_length = 25
productname_max_length = 90
productdescription_max_length = 500
brand_max_length = 25


nb_triplets_query_product_buckets = 32768

def exportToPickle(inputTFModel, outputFile):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
        saver = tf.train.import_meta_graph(inputTFModel + ".meta")
        saver.restore(session, inputTFModel)
        graph = tf.get_default_graph()
        W = tf.trainable_variables()[0]
        values = W.eval()
        b = tf.trainable_variables()[1]
        exportToProto(modelname, modelname + '.proto')
        values_b = b.eval()
        weights = np.append(values, values_b);
        np.save(outputFile, weights, allow_pickle=True)

def exportToProto(inputTFModel, outputFile):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
        saver = tf.train.import_meta_graph(inputTFModel + ".meta")
        saver.restore(session, inputTFModel)
        graph = tf.get_default_graph()
        W = tf.trainable_variables()[0]
        values = W.eval()
        b = tf.trainable_variables()[1]
        values_b = b.eval()
        weights = np.append(values, values_b);
        proto_weights = tf.train.Example(features=tf.train.Features(feature={
            'weigthts': tf.train.Feature(float_list=tf.train.FloatList(value=weights))}))
        serialized_proto_weights = proto_weights.SerializeToString()
        writer = tf.python_io.TFRecordWriter(outputFile)
        writer.write(serialized_proto_weights)

def exportToJson(inputTFModel, outputFile):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
        saver = tf.train.import_meta_graph(inputTFModel + ".meta")
        saver.restore(session, inputTFModel)
        graph = tf.get_default_graph()
        W = tf.trainable_variables()[0]
        values = W.eval()
        b = tf.trainable_variables()[1]
        values_b = b.eval()
        weights = np.append(values, values_b);
        model_weights = tf.train.Example(features=tf.train.Features(feature={
            'weigthts': tf.train.Feature(float_list=tf.train.FloatList(value=weights))}))

        final_weights = {"weights": [model_weights["0"][str(i)] for i in range(len(model_weights.values()[0]))]}
        final_output = open(outputFile, "w")
        final_output.write(json.dumps(final_weights))
        final_output.close()


def loadModelProto(modelFile, size):
    for serialized_example_validation in tf.python_io.tf_record_iterator(modelFile):
        serialized_batch_validation = tf.train.Example()
        serialized_batch_validation.ParseFromString(serialized_example_validation)

        weights_index = np.array(serialized_batch_validation.features.feature['weigthts'].float_list.value)
        #weights = np.zeros(size)
        #weights[weights_index]=1
        return weights_index


def loadModelPickel(modelFile):
    return np.load(modelFile);

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


def query2producttripletrepresentation(query, product, len, BUCKETS):
    features = Set()
    qgrams_4 = getTriplets(query, len)
    pgrams_4 = getTriplets(product, len)
    for gram in qgrams_4:
        if gram in pgrams_4:
            features.add(abs(int(mmh3.hash(gram))) % BUCKETS)
    return features


def getFeatures(query, productname, productbrand):
    query_productname_triplets = np.zeros(nb_triplets_query_product_buckets)
    query_brand_triplets = np.zeros(nb_triplets_query_product_buckets)
    query_productname_brand_triplets = np.zeros(nb_triplets_query_product_buckets)


    querynormalized = normalizeQuery(query,query_max_length)
    productnamenormalized = normalizeProduct(productname, productname_max_length)
    productbrandnormalized = normalizeProduct(productbrand, brand_max_length)



    features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, 3,
                                                                       nb_triplets_query_product_buckets)

    features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized, 3,
                                                                        nb_triplets_query_product_buckets)

    features_productname_brand_triplets = features_productname_triplets.intersection(features_productbrand_triplets)


    for cidx in features_productname_triplets:
        query_productname_triplets[cidx] = 1
    for cidx in features_productbrand_triplets:
        query_brand_triplets[cidx] = 1
    for cidx in features_productname_brand_triplets:
        query_productname_brand_triplets[cidx] = 1

    result = query_productname_triplets;
    result = np.append(result,query_brand_triplets)
    result = np.append(result,query_productname_brand_triplets)
    result = np.append(result,[1]) #bias
    return result


def sigmoid(features,weight):
    return 1/(1+np.exp(-np.dot(features,weight)))

def score(model, query, pname, pbrand):
    features = getFeatures(query,pname,pbrand)
    return sigmoid(features, model)

def toJson(model_weights, filename):
    final_weights = {"weights": model_weights.tolist()}
    final_output = open(filename, "w")
    final_output.write(json.dumps(final_weights))
    final_output.close()



modelname = sys.argv[1]
#toJson(modelname, modelname+'.')
#exportToPickle(modelname,modelname+'export')
#model=loadModelPickel(modelname+'export.npy')
model=loadModelProto(modelname+'.proto',nb_triplets_query_product_buckets + 1)
toJson(model, modelname+'.json')
print(score(model,"excedrin migrain","100ct ach caplet excedrin174  head pain reliev tension","excedrin"))
print(score(model,"vaccum shark cleaner","vaccum dyson","dyson"))



print(score(model,"breast chicken","184 broth campbellaposs174 chicken homestyle153 italianstyl meatbal oz soup spinach wed","campbel"))



