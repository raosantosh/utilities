from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import mmh3
import time
from sets import Set

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
access_mode = "r"

#
# paths to filess

cwd = os.getcwd()
path_data = cwd+'/../resources/datasets/'
path_model = cwd+'/../resources/datasets/models/'
path_annotations = cwd + '/../resources/datasets/annotations/'


positives_training_file = path_data+"positive_training_samples_query_productname_descr_brand_category_stemmed_unique_131.csv"
negatives_training_file = path_data+"negative_training_samples_query_productname_descr_brand_categgory_stemmed_unique_131.csv"
test_positives_file = path_data+"test_positives_unique_file_131"
test_negatives_file = path_data+"test_negatives_unique_file_131"

validation_file = path_annotations + 'annotations_courtney.csv'


max_positive_training_samples_size = 139404 - 9404
max_negative_training_samples_size = 407449 - 7449
nb_test_batches = 10

p_fptr = open(positives_training_file, mode=access_mode)
n_fptr = open(negatives_training_file, mode=access_mode)

fp = open(test_positives_file, access_mode)
fn = open(test_negatives_file, access_mode)
vp = open(validation_file, access_mode)

brand_dict = {}
cateory_dict = {}
brand_dict_readindex = {}
cateory_dict_readindex = {}


batch_positives = 4

negative_samples_factor = 7*3

positive_index = 0
negative_index = 0

query_max_length = 25
productname_max_length = 90
productdescription_max_length = 500
brand_max_length = 25


nb_triplets_query_product_buckets = 32768

batch_size = batch_positives * (negative_samples_factor + 1)
miniBatchDisplayFreq = 100

QUERY_INDEX = 0
PRODCUTNAME_INDEX_POSITIVE = 1
PRODCUTNAME_INDEX_NEGATIVE = 0

PRODUCTDESC_INDEX_POSITIVE = 2
PRODUCTDESC_INDEX_NEGATIVE = 1

PRODUCTBRAND_INDEX_POSITIVE = 3
PRODUCTBRAND_INDEX_NEGATIVE = 2

PRODUCTCATEGORY_INDEX_POSITIVE = 4
PRODUCTCATEGORY_INDEX_NEGATIVE = 3
PRODUCTLABEL_INDEX_POSITIVE = 5


droupout_rate = 0.4

def loadBrandCategoryDict(inputfile):

    p_fptr = open(inputfile, mode=access_mode)
    for index in range(1,max_positive_training_samples_size-1):
        p_line = p_fptr.readline()
        ntokens = p_line.rstrip().split('|')
        if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) != 0:
            if ntokens[PRODUCTBRAND_INDEX_NEGATIVE] not in brand_dict:
                brand_dict[ntokens[PRODUCTBRAND_INDEX_NEGATIVE]]=[]
                brand_dict_readindex[ntokens[PRODUCTBRAND_INDEX_NEGATIVE]]=0
            if len(brand_dict[ntokens[PRODUCTBRAND_INDEX_NEGATIVE]])<100:
                brand_dict[ntokens[PRODUCTBRAND_INDEX_NEGATIVE]].append(p_line)


        if len(ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE]) != 0:
            if ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE] not in brand_dict:
                cateory_dict[ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE]]=[]
                cateory_dict_readindex[ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE]]=0
            if len(cateory_dict[ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE]])<100:
                cateory_dict[ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE]].append(p_line)



    p_fptr.close()
    #return brand_dict, cateory_dict, brand_dict_readindex, cateory_dict_readindex

def getNextCategorySample(category):
    if category not in cateory_dict:
        return ""
    line =  cateory_dict[category][cateory_dict_readindex[category]]
    cateory_dict_readindex[category] = (cateory_dict_readindex[category]+1)%len(cateory_dict[category])
    return line



def getNextBrandSample(brand):
    if brand not in brand_dict:
        return ""
    line =  brand_dict[brand][brand_dict_readindex[brand]]
    brand_dict_readindex[brand] = (brand_dict_readindex[brand]+1)%len(brand_dict[brand])
    return line


def getBrandOfSample(line):
    ntokens = line.rstrip().split('|')
    return ntokens[PRODUCTBRAND_INDEX_NEGATIVE]

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


def get_next_batch():

    labels = np.zeros((batch_size, 1))
    query_productname_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_productdescription_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))

    query_productname_description_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_productdescription_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))
    query_productname_description_brand_triplets = np.zeros((batch_size, nb_triplets_query_product_buckets, 1))

    product_index = 0

    for index in range(batch_positives):
        positives_line = read_positives_file()
        ptokens = positives_line.rstrip().split('|')

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;


        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productdescriptionnormalized = ""
        productbrandnormalized = ""
        if len(ptokens[PRODUCTDESC_INDEX_POSITIVE]) != 0:
            productdescriptionnormalized = normalizeProduct(ptokens[PRODUCTDESC_INDEX_POSITIVE], productdescription_max_length)

        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)

        features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, 3,nb_triplets_query_product_buckets)
        features_productdecription_triplets = query2producttripletrepresentation(querynormalized, productdescriptionnormalized, 4,nb_triplets_query_product_buckets)
        features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized, 3, nb_triplets_query_product_buckets)
        features_productname_description_triplets  = features_productname_triplets.intersection(features_productdecription_triplets)
        features_productname_brand_triplets  = features_productname_triplets.intersection(features_productbrand_triplets)
        features_productdescription_brand_triplets  = features_productdecription_triplets.intersection(features_productname_brand_triplets)
        features_productname_description_brand_triplets = features_productname_brand_triplets.intersection(features_productdecription_triplets)

        for cidx in features_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            query_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productdecription_triplets:
            query_productdescription_triplets[product_index, cidx, 0] = 1

        for cidx in features_productname_description_triplets:
            query_productname_description_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_brand_triplets:
            query_productname_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productdescription_brand_triplets:
            query_productdescription_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_description_brand_triplets:
            query_productname_description_brand_triplets[product_index, cidx, 0] = 1

        product_index += 1
        negatives = 0

        #negative random sampling
        trial_cat_sampling = 0
        trial_brand_sampling = 0
        while (negatives <   negative_samples_factor):
            #print(str(product_index)+" "+str(negatives))
            if(negatives%3)==0:
                if trial_cat_sampling==5:
                    #we have not been able to find a product of
                    #different brand in this category after 5 trials
                    #print("T "+str(product_index) + " " + str(negatives))
                    trial_cat_sampling=0
                    negatives+=1
                    #print("5 trials..inc neg index "+str(product_index)+" "+str(negatives))
                    continue
                trial_cat_sampling+=1
                category = ptokens[PRODUCTCATEGORY_INDEX_POSITIVE]
                if (len(category) == 0):
                    continue
                negatives_line = getNextCategorySample(category)
                ntokens = negatives_line.rstrip().split('|')
                if len(ntokens)<PRODUCTBRAND_INDEX_NEGATIVE+1:
                    continue
                if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) == 0:
                    continue
                if len(ptokens)<PRODUCTBRAND_INDEX_POSITIVE+1:
                    continue
                if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) == 0:
                    continue
                if ntokens[PRODUCTBRAND_INDEX_NEGATIVE] == ptokens[PRODUCTBRAND_INDEX_POSITIVE]:
                    continue
                trial_cat_sampling=0
                #print('successful cat neg sampling category:'+category+'\t'+ntokens[PRODUCTBRAND_INDEX_NEGATIVE]+'\t'+ptokens[PRODUCTBRAND_INDEX_POSITIVE])
                # we have smapled a brand from the same category but different brand
            elif(negatives%3)==1:
                if trial_brand_sampling==5:
                    #we have not been able to find a product of
                    #different brand in this category after 5 trials
                    #print("T "+str(product_index) + " " + str(negatives))
                    trial_brand_sampling=0
                    negatives+=1
                    #print("5 trials..inc neg index "+str(product_index)+" "+str(negatives))
                    continue
                trial_brand_sampling+=1
                brand = ptokens[PRODUCTBRAND_INDEX_POSITIVE]
                if (len(brand) == 0):
                    continue
                negatives_line = getNextBrandSample(brand)
                ntokens = negatives_line.rstrip().split('|')
                if len(ntokens)<PRODUCTCATEGORY_INDEX_NEGATIVE+1:
                    continue
                if len(ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE]) == 0:
                    continue
                if len(ptokens)<PRODUCTCATEGORY_INDEX_POSITIVE+1:
                    continue
                if len(ptokens[PRODUCTCATEGORY_INDEX_POSITIVE]) == 0:
                    continue
                if ntokens[PRODUCTCATEGORY_INDEX_NEGATIVE] == ptokens[PRODUCTCATEGORY_INDEX_POSITIVE]:
                    continue
                trial_brand_sampling=0
                #print('successful cat neg sampling category:'+category+'\t'+ntokens[PRODUCTBRAND_INDEX_NEGATIVE]+'\t'+ptokens[PRODUCTBRAND_INDEX_POSITIVE])
                # we have smapled a brand from the same category but different brand

            else:
                #print('successful normal sampling '+str(product_index)+" "+str(negatives))
                negatives_line = read_negatives_file()
                ntokens = negatives_line.rstrip().split('|')


            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                #print("no product name")
                continue;

            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            productdescriptionnormalized = ""
            productbrandnormalized = ""

            if len(ntokens[PRODUCTDESC_INDEX_NEGATIVE]) != 0:
                productdescriptionnormalized = normalizeProduct(ntokens[PRODUCTDESC_INDEX_NEGATIVE],
                                                            productdescription_max_length)
            if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) != 0:
                productbrandnormalized = normalizeProduct(ntokens[PRODUCTBRAND_INDEX_NEGATIVE], brand_max_length)

            features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, 3,nb_triplets_query_product_buckets)
            features_productdecription_triplets = query2producttripletrepresentation(querynormalized,
                                                                                     productdescriptionnormalized,4,
                                                                                     nb_triplets_query_product_buckets)
            features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized,
                                                                                3,nb_triplets_query_product_buckets)

            features_productname_description_triplets = features_productname_triplets.intersection(
                features_productdecription_triplets)
            features_productname_brand_triplets = features_productname_triplets.intersection(
                features_productbrand_triplets)

            features_productdescription_brand_triplets = features_productdecription_triplets.intersection(
                features_productname_brand_triplets)
            features_productname_description_brand_triplets = features_productname_brand_triplets.intersection(
                features_productdecription_triplets)



            for cidx in features_productname_triplets:
                query_productname_triplets[product_index, cidx, 0] = 1
            for cidx in features_productbrand_triplets:
                query_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productdecription_triplets:
                query_productdescription_triplets[product_index, cidx, 0] = 1

            for cidx in features_productname_description_triplets:
                query_productname_description_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_brand_triplets:
                query_productname_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productdescription_brand_triplets:
                query_productdescription_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_description_brand_triplets:
                query_productname_description_brand_triplets[product_index, cidx, 0] = 1


            product_index += 1
            negatives += 1
            #print("increment product index / neg index "+str(product_index)+" "+str(negatives))


    for index in range(batch_size):
        if index % (negative_samples_factor + 1) == 0:
            labels[index, 0] = 1  # labels[index] = 1

    return labels, query_productname_triplets, query_productdescription_triplets, query_brand_triplets,\
           query_productname_description_triplets, query_productname_brand_triplets, \
           query_productdescription_brand_triplets, query_productname_description_brand_triplets


def get_next_validation_data(validation_file, validations_count):


    global vp
    test_data_size = validations_count

    test_negative_samples_factor = negative_samples_factor
    test_labels = np.zeros((test_data_size, 1))

    query_productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productdescription_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    query_productname_description_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productdescription_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_description_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    product_index = 0

    for index in range(validations_count):
        positives_line = vp.readline()
        if not positives_line:
            vp.close()
            vp = open(validation_file, 'r')
            positives_line = vp.readline()

        ptokens = positives_line.rstrip().split('\t')
        if len(ptokens)<PRODUCTLABEL_INDEX_POSITIVE:
            continue;
        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;

        if ptokens[PRODUCTLABEL_INDEX_POSITIVE] == 'N/A':
            continue;

        if ptokens[PRODUCTLABEL_INDEX_POSITIVE] == 'Good':
            test_labels[product_index, 0] = 1
        else:
            test_labels[product_index, 0] = 0

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;

        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productdescriptionnormalized = ""
        productbrandnormalized = ""

        if len(ptokens[PRODUCTDESC_INDEX_POSITIVE]) != 0:
            productdescriptionnormalized = normalizeProduct(ptokens[PRODUCTDESC_INDEX_POSITIVE], productdescription_max_length)

        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)

        features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, 3,nb_triplets_query_product_buckets)
        features_productdecription_triplets = query2producttripletrepresentation(querynormalized, productdescriptionnormalized, 4,nb_triplets_query_product_buckets)
        features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized,3,  nb_triplets_query_product_buckets)
        features_productname_description_triplets  = features_productname_triplets.intersection(features_productdecription_triplets)
        features_productname_brand_triplets  = features_productname_triplets.intersection(features_productbrand_triplets)
        features_productdescription_brand_triplets  = features_productdecription_triplets.intersection(features_productname_brand_triplets)
        features_productname_description_brand_triplets = features_productname_brand_triplets.intersection(features_productdecription_triplets)

        for cidx in features_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            query_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productdecription_triplets:
            query_productdescription_triplets[product_index, cidx, 0] = 1

        for cidx in features_productname_description_triplets:
            query_productname_description_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_brand_triplets:
            query_productname_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productdescription_brand_triplets:
            query_productdescription_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_description_brand_triplets:
            query_productname_description_brand_triplets[product_index, cidx, 0] = 1




        product_index += 1



    return test_labels, query_productname_triplets, query_productdescription_triplets, query_brand_triplets, \
           query_productname_description_triplets, query_productname_brand_triplets, \
           query_productdescription_brand_triplets, query_productname_description_brand_triplets


def get_next_test_data(pos_test_file_name, neg_test_file_name, positives_count):

    global fn
    global fp
    test_negative_samples_factor = negative_samples_factor
    test_data_size = positives_count * (test_negative_samples_factor + 1)
    test_labels = np.zeros((test_data_size, 1))

    query_productname_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productdescription_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    query_productname_description_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productdescription_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))
    query_productname_description_brand_triplets = np.zeros((test_data_size, nb_triplets_query_product_buckets, 1))

    product_index = 0

    for index in range(positives_count):
        positives_line = fp.readline()
        if not positives_line:
            fp.close()
            fp = open(pos_test_file_name, 'r')
            positives_line = fp.readline()

        ptokens = positives_line.rstrip().split('|')

        if (len(ptokens[QUERY_INDEX]) == 0 or len(ptokens[PRODCUTNAME_INDEX_POSITIVE]) == 0):
            continue;

        querynormalized = normalizeQuery(ptokens[QUERY_INDEX],query_max_length)
        productnamenormalized = normalizeProduct(ptokens[PRODCUTNAME_INDEX_POSITIVE], productname_max_length)
        productdescriptionnormalized = ""
        productbrandnormalized = ""

        if len(ptokens[PRODUCTDESC_INDEX_POSITIVE]) != 0:
            productdescriptionnormalized = normalizeProduct(ptokens[PRODUCTDESC_INDEX_POSITIVE], productdescription_max_length)

        if len(ptokens[PRODUCTBRAND_INDEX_POSITIVE]) != 0:
            productbrandnormalized =  normalizeProduct(ptokens[PRODUCTBRAND_INDEX_POSITIVE], brand_max_length)

        features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized, 3,nb_triplets_query_product_buckets)
        features_productdecription_triplets = query2producttripletrepresentation(querynormalized, productdescriptionnormalized, 4,nb_triplets_query_product_buckets)
        features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized,3,  nb_triplets_query_product_buckets)
        features_productname_description_triplets  = features_productname_triplets.intersection(features_productdecription_triplets)
        features_productname_brand_triplets  = features_productname_triplets.intersection(features_productbrand_triplets)
        features_productdescription_brand_triplets  = features_productdecription_triplets.intersection(features_productname_brand_triplets)
        features_productname_description_brand_triplets = features_productname_brand_triplets.intersection(features_productdecription_triplets)

        for cidx in features_productname_triplets:
            query_productname_triplets[product_index, cidx, 0] = 1
        for cidx in features_productbrand_triplets:
            query_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productdecription_triplets:
            query_productdescription_triplets[product_index, cidx, 0] = 1

        for cidx in features_productname_description_triplets:
            query_productname_description_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_brand_triplets:
            query_productname_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productdescription_brand_triplets:
            query_productdescription_brand_triplets[product_index, cidx, 0] = 1
        for cidx in features_productname_description_brand_triplets:
            query_productname_description_brand_triplets[product_index, cidx, 0] = 1




        product_index += 1
        negatives = 0

        while (negatives != test_negative_samples_factor):
            negatives_line = fn.readline()
            if not negatives_line:
                fn.close()
                fn = open(neg_test_file_name, 'r')
                negatives_line = fn.readline()

            ntokens = negatives_line.rstrip().split('|')
            if (len(ntokens[PRODCUTNAME_INDEX_NEGATIVE]) == 0):
                continue;

            productnamenormalized = normalizeProduct(ntokens[PRODCUTNAME_INDEX_NEGATIVE], productname_max_length)
            productdescriptionnormalized = ""
            productbrandnormalized = ""


            if len(ntokens[PRODUCTDESC_INDEX_NEGATIVE]) != 0:
                productdescriptionnormalized = normalizeProduct(ntokens[PRODUCTDESC_INDEX_NEGATIVE],
                                                                productdescription_max_length)
            if len(ntokens[PRODUCTBRAND_INDEX_NEGATIVE]) != 0:
                productbrandnormalized = normalizeProduct(ntokens[PRODUCTBRAND_INDEX_NEGATIVE], brand_max_length)

            features_productname_triplets = query2producttripletrepresentation(querynormalized, productnamenormalized,
                                                                               3, nb_triplets_query_product_buckets)
            features_productdecription_triplets = query2producttripletrepresentation(querynormalized,
                                                                                     productdescriptionnormalized,
                                                                                     4, nb_triplets_query_product_buckets)
            features_productbrand_triplets = query2producttripletrepresentation(querynormalized, productbrandnormalized,
                                                                                3, nb_triplets_query_product_buckets)

            features_productname_description_triplets = features_productname_triplets.intersection(
                features_productdecription_triplets)
            features_productname_brand_triplets = features_productname_triplets.intersection(
                features_productbrand_triplets)

            features_productdescription_brand_triplets = features_productdecription_triplets.intersection(
                features_productname_brand_triplets)
            features_productname_description_brand_triplets = features_productname_brand_triplets.intersection(
                features_productdecription_triplets)

            for cidx in features_productname_triplets:
                query_productname_triplets[product_index, cidx, 0] = 1
            for cidx in features_productbrand_triplets:
                query_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productdecription_triplets:
                query_productdescription_triplets[product_index, cidx, 0] = 1

            for cidx in features_productname_description_triplets:
                query_productname_description_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_brand_triplets:
                query_productname_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productdescription_brand_triplets:
                query_productdescription_brand_triplets[product_index, cidx, 0] = 1
            for cidx in features_productname_description_brand_triplets:
                query_productname_description_brand_triplets[product_index, cidx, 0] = 1

            product_index += 1
            negatives += 1

    for index in range(test_data_size):
        if index % (test_negative_samples_factor + 1) == 0:
            test_labels[index, 0] = 1  # test_labels[index] = 1

    return test_labels, query_productname_triplets, query_productdescription_triplets, query_brand_triplets, \
           query_productname_description_triplets, query_productname_brand_triplets, \
           query_productdescription_brand_triplets, query_productname_description_brand_triplets


def phi(x, n_output, droupout_rate, isTraining, name=None, activation=None, reuse=None,
        dropout=None):


    n_input = x.get_shape().as_list()[1]


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
    query_productname_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                    name="query_productname_triplets")
    query_productname_triplets_emb_flat = tf.reshape(query_productname_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                     name="query_productname_triplets_flat")

    query_productdescription_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                    name="query_productdesc_triplets")
    query_productdescription_triplets_emb_flat = tf.reshape(query_productdescription_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                     name="query_productdesc_triplets_flat")


    query_productbrand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                    name="query_productbrand_triplets")
    query_productbrand_triplets_emb_flat = tf.reshape(query_productbrand_triplets_emb, [-1, nb_triplets_query_product_buckets],
                                                     name="query_productbrand_triplets_flat")

    query_productname_description_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="query_productname_description_triplets")
    query_productname_description_triplets_emb_flat = tf.reshape(query_productname_description_triplets_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="query_productname_description_triplets_flat")

    query_productname_brand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="query_productname_brand_triplets")
    query_productname_brand_triplets_emb_flat = tf.reshape(query_productname_brand_triplets_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="query_productname_brand_triplets_flat")
    query_productdescription_brand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="query_productdescription_brand_triplets")
    query_productdescription_brand_triplets_emb_flat = tf.reshape(query_productdescription_brand_triplets_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="query_productdescription_brand_triplets_flat")

    query_productname_description_brand_triplets_emb = tf.placeholder(tf.float32, [None, nb_triplets_query_product_buckets, 1],
                                                     name="query_productname_description_brand_triplets")
    query_productname_description_brand_triplets_emb_flat = tf.reshape(query_productname_description_brand_triplets_emb,
                                                      [-1, nb_triplets_query_product_buckets],
                                                      name="query_productname_description_brand_triplets_flat")

    query_product = tf.concat([query_productname_triplets_emb_flat, query_productdescription_triplets_emb_flat,
                               query_productbrand_triplets_emb_flat, query_productname_description_triplets_emb_flat,
                               query_productname_brand_triplets_emb_flat,query_productdescription_brand_triplets_emb_flat,
                               query_productname_description_brand_triplets_emb_flat], 1)

    query_product_out_2, query_product_out_wt_2 = phi(query_product, n_output=1, droupout_rate=droupout_rate,
                                                      activation=None, name='query_fc_layer_2',
                                                      isTraining=isTrain,
                                                      dropout=True)

    y_true = tf.placeholder(tf.float32, shape=(None, 1))
    y_prediction = query_product_out_2

    pCTR = tf.nn.sigmoid(y_prediction, name='pCTR')
    #cross_entropy_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_prediction)
    #cross_entropy_tensor = tf.concat([cross_entropy_tensor, cross_entropy_tensor], 1)
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

loadBrandCategoryDict(negatives_training_file)

with tf.Session() as session:
    session.run(init)
    #session.run(reset_metrics)

    #session.run(init_local)
    for iteration_index in range(max_iterations):

        for batch_index in range(1,num_batches+1):

            batch_labels, query_productname_triplets_batch_data, \
                    query_productdescription_triplets_batch_data ,\
                        query_productbrand_triplets_batch_data, \
                        query_productname_description_triplets_batch_data, \
                        query_productname_brand_triplets_batch_data, \
                        query_productdescription_brand_triplets_batch_data, \
                        query_productname_description_brand_triplets_batch_data = get_next_batch()


            # if batch_index % 100 == 0:
            #     test_labels, test_query_productname_triplets, \
            #         test_query_productdescription_triplets_batch_data,  \
            #             test_query_productbrand_triplets_batch_data , \
            #     test_query_productname_description_triplets_batch_data, test_query_productname_brand_triplets_batch_data, \
            #     test_query_productdescription_brand_triplets_batch_data , \
            #     test_query_productname_description_brand_triplets_batch_data= get_next_test_data(test_positives_file,
            #                                                                         test_negatives_file, 100)
            #
            #     fd_test = {y_true: test_labels, query_productname_triplets_emb: test_query_productname_triplets,
            #                query_productdescription_triplets_emb: test_query_productdescription_triplets_batch_data,
            #                query_productbrand_triplets_emb:  test_query_productbrand_triplets_batch_data,
            #                query_productname_description_triplets_emb: test_query_productname_description_triplets_batch_data,
            #                query_productname_brand_triplets_emb: test_query_productname_brand_triplets_batch_data,
            #                query_productdescription_brand_triplets_emb:  test_query_productdescription_brand_triplets_batch_data,
            #                query_productname_description_brand_triplets_emb: test_query_productname_description_brand_triplets_batch_data,
            #                isTrain: False}
            #
            #
            #     _, _, _, _ = session.run(
            #         [validation_cross_entropy_metric, pCTR, validation_auc_op, validation_auc], feed_dict=fd_test)


            if batch_index % 1000 == 0:
                auc_val=0
                avgloss=0
                auc_val2=0
                #for loop in range(nb_test_batches):
                test_labels, test_query_productname_triplets, \
                test_query_productdescription_triplets_batch_data, \
                test_query_productbrand_triplets_batch_data, \
                test_query_productname_description_triplets_batch_data, test_query_productname_brand_triplets_batch_data, \
                test_query_productdescription_brand_triplets_batch_data, \
                test_query_productname_description_brand_triplets_batch_data = get_next_validation_data(validation_file, 502)

                fd_test = {y_true: test_labels, query_productname_triplets_emb: test_query_productname_triplets,
                           query_productdescription_triplets_emb: test_query_productdescription_triplets_batch_data,
                           query_productbrand_triplets_emb: test_query_productbrand_triplets_batch_data,
                           query_productname_description_triplets_emb: test_query_productname_description_triplets_batch_data,
                           query_productname_brand_triplets_emb: test_query_productname_brand_triplets_batch_data,
                           query_productdescription_brand_triplets_emb: test_query_productdescription_brand_triplets_batch_data,
                           query_productname_description_brand_triplets_emb: test_query_productname_description_brand_triplets_batch_data,
                           isTrain: False}


                loss, y_score = session.run(
                    [cross_entropy, pCTR],
                    feed_dict=fd_test)

                #auc_val +=  session.run(validation_auc) #roc_auc_score(test_labels, y_score)
                auc_val2 = roc_auc_score(test_labels, y_score)
                avgloss=loss

                #auc_val = auc_val / nb_test_batches
                #avgloss = avgloss / nb_test_batches
                #auc_val2 = auc_val2 / nb_test_batches

                end = time.time()
                print('iteration ' + str(iteration_index) + ' auc@test  ' + str(auc_val2) +  ' loss@test '+ str(avgloss)  +' done ' + str(end - start))
                start = time.time()
                #session.run(reset_metrics)
                saver.save(session, path_model+'/'+modelname,global_step=num_batches * (iteration_index + 1))

            if batch_index % 100 == 0:
                end = time.time()
                training_loss = session.run(cross_entropy, feed_dict={y_true: batch_labels,
                                                                      query_productname_triplets_emb: query_productname_triplets_batch_data,
                                                                      query_productdescription_triplets_emb: query_productdescription_triplets_batch_data,
                                                                      query_productbrand_triplets_emb: query_productbrand_triplets_batch_data,
                                                                      query_productname_description_triplets_emb: query_productname_description_triplets_batch_data,
                                                                      query_productname_brand_triplets_emb: query_productname_brand_triplets_batch_data,
                                                                      query_productdescription_brand_triplets_emb: query_productdescription_brand_triplets_batch_data,
                                                                      query_productname_description_brand_triplets_emb: query_productname_description_brand_triplets_batch_data,
                                                                      isTrain: False})

                print('iteration ' + str(iteration_index) + ' batch ' + str(batch_index + 1) + ' loss ' + str(
                    training_loss) + ' done ' + str(end - start))

                start = time.time()
            else:
                session.run(adam_train_step, feed_dict={y_true: batch_labels,
                                                                      query_productname_triplets_emb: query_productname_triplets_batch_data,
                                                                      query_productdescription_triplets_emb: query_productdescription_triplets_batch_data,
                                                                      query_productbrand_triplets_emb: query_productbrand_triplets_batch_data,
                                                                      query_productname_description_triplets_emb: query_productname_description_triplets_batch_data,
                                                                      query_productname_brand_triplets_emb: query_productname_brand_triplets_batch_data,
                                                                      query_productdescription_brand_triplets_emb: query_productdescription_brand_triplets_batch_data,
                                                                      query_productname_description_brand_triplets_emb: query_productname_description_brand_triplets_batch_data,
                                                                      isTrain: True})

if (~p_fptr.closed):
    p_fptr.close()

if (~n_fptr.closed):
    n_fptr.close()

if (~fp.closed):
    fp.close()

if (~fn.closed):
    fn.close()
