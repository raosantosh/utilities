import argparse
import logging
import tensorflow as tf
import numpy as np
import sys
import pickle

# Parameters from grid
parser = argparse.ArgumentParser(
    description='Learn one classifier on each language, starting from a sparse representation of products')

parser.add_argument('--catalog_id', type=str)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[DistanceMatrix]")


logger.info("Starting job (catalog_id={})...".format(args.catalog_id))

path_data = '/var/opt/amin/Data/datasets/'

batch_positives = 4
negative_samples_factor = 7
nb_triplets_query_product_buckets = 2 ** 13
batch_size = batch_positives * (negative_samples_factor + 1)


def extract_data(serialized_data):
    if serialized_data is None:
        return None, None, None

    serialized_batch = tf.train.Example()
    serialized_batch.ParseFromString(serialized_data)

    # Extracting labels
    batch_labels = np.array(serialized_batch.features.feature['label'].int64_list.value).reshape(batch_size, 1)

    # Extracting queries
    queries_indices = np.array(serialized_batch.features.feature['query'].int64_list.value)
    batch_queries = np.zeros((batch_size * nb_triplets_query_product_buckets))
    batch_queries[queries_indices] = 1
    batch_queries = batch_queries.reshape(batch_size, nb_triplets_query_product_buckets, 1)

    # Extracting products
    products_indices = np.array(serialized_batch.features.feature['product'].int64_list.value)
    batch_products = np.zeros((batch_size * nb_triplets_query_product_buckets))
    batch_products[products_indices] = 1
    batch_products = batch_products.reshape(batch_size,
                                            nb_triplets_query_product_buckets, 1)

    return batch_queries, batch_products, batch_labels


data_iterator = tf.python_io.tf_record_iterator(path_data + "positive_negative_{}.proto".format(str(args.catalog_id)))

global_sum = np.zeros((8192, 1))

for batch_idx, batch in enumerate(data_iterator):

    if batch_idx % 1000 == 0:
        logger.info("Processing batch {}".format(batch_idx))

    _, products, _ = extract_data(batch)

    product_sum = np.sum(products, axis=0)

    global_sum += product_sum


with open("triplets_{}.pickle".format(args.catalog_id), "wb") as fw:
    pickle._dump(global_sum, fw)

logger.info("Done")
