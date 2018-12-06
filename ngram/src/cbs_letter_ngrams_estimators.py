from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

cwd = '/home/a.mantrach/LettersQuerySkuModel/resources/'
log_location = cwd + 'logs/'
path_data = cwd+'datasets/'
path_model = cwd + 'models/'
positives_training_file = path_data + "positive_training_samples_query_productname_131.csv"
negatives_training_file = path_data + "negative_training_samples_query_productname_131.csv"

QUERY_INDEX = 2
PRODCUTNAME_INDEX_POSITIVE = 3
PRODUCTBRAND_INDEX_POSITIVE = 4
PRODCUTNAME_INDEX_NEGATIVE = 1
PRODUCTBRAND_INDEX_NEGATIVE = 2


separator = '|'



def positive_input_fn(positives_training_file):

    dataset_positive = tf.data.TextLineDataset(positives_training_file)
    dataset_positive = dataset_positive.map(
        lambda line: tf.string_split(line, delimiter=separator, skip_empty=False).
                         values[QUERY_INDEX:PRODUCTBRAND_INDEX_POSITIVE+1])


    def parser_word2grams(word):
        tf.tile([word], [tf.size(tf.string_split([word], delimiter="", skip_empty=False))])

    def parser_word2grams(word):
        word_length = tf.size(tf.string_split([word[0]], delimiter="", skip_empty=False))
        range_word_length = tf.range(0,word_length-2,1)
        multiple_words= tf.tile([word],word_length-2)
        tf.map_fn(parser_word2grams,word_length,back_prop=False)



    def parser_positif(record):

        words_query=tf.string_split([record[0]],delimiter=" ", skip_empty=True)
        words_pname=tf.string_split([record[1]],delimiter=" ", skip_empty=True)
        words_brand=tf.string_split([record[2]],delimiter=" ", skip_empty=True)

        productname_length = tf.size(tf.string_split([record[1]], delimiter="", skip_empty=False))
        brand_length = tf.size(tf.string_split([record[2]], delimiter="", skip_empty=False))

        tf.map_fn(parser_word2grams,words_query,back_prop=False)

        range_productname = tf.range(0,productname_length,1)
        range_brand_length = tf.range(0,brand_length,1)

        tf.substr(record[0],0,3)
        return {"query": record[0], "productname": record[1], "brand": record[2], "label": tf.ones(record.shape[0])}

    dataset_positive = dataset_positive.map(parser_positif)
    dataset_positive = dataset_positive.batch(1)
    dataset_positive = dataset_positive.repeat(5)
    iterator = dataset_positive.make_one_shot_iterator()



def input_fn(dataset_positive, dataset_negative):
   return feature_dict, label


tf.string_split(tf.constant('hello the world'),delimiter=' ',skip_empty=False)