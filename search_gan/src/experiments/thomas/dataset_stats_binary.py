import argparse
import numpy as np
import tensorflow as tf
import logging


# Training - validation files location
path_data = '/var/opt/amin/Data/datasets/'

# Parameters from grid
parser = argparse.ArgumentParser(
    description='Compute stats on the datasets')

parser.add_argument('--source_id_and_target_id', type=str)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[Dataset Stats]")

source_id, target_id = args.source_id_and_target_id.split("_")

logger.info("Starting job (source={}, target={})...".format(source_id, target_id))

# Static parameters
batch_positives = 4
negative_samples_factor = 7
nb_triplets_query_product_buckets = 2 ** 13
batch_size = batch_positives * (negative_samples_factor + 1)
batch_size_test = 32  # 640


def flatten_and_reshape(input_x):
    x_flat = tf.reshape(input_x,
                        shape=[-1, nb_triplets_query_product_buckets],
                        name="{}_flat".format(input_x.name.split(":")[0]))
    return tf.concat([x_flat], 1)


def parse_serialized_batch(serialized_data, batch_size):
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


class CircularRecordIterator(object):

    def __init__(self, catalog_id, mode):
        # Saving data path
        self._data_path = path_data + "positive_negative_1p_7n_32_partner{}_{}.proto".format(str(catalog_id), mode)
        # Initializing iterator
        self._iterator = self._get_iterator()

    def _get_iterator(self):
        return tf.python_io.tf_record_iterator(self._data_path)

    def get_next_serialized_batch(self):
        """
        :return: (next_batch_serialized, isNewEpoch)
        """
        serialized_data = next(self._iterator, None)

        if serialized_data is not None:
            return serialized_data, False

        # Otherwise, we have reached the end of the dataset
        # We reinit the iterator
        self._iterator = self._get_iterator()
        return next(self._iterator), True


iterators = {
    "train": {
        "source": CircularRecordIterator(source_id, "train"),
        "target": CircularRecordIterator(target_id, "train"),
        "dans_source": CircularRecordIterator(source_id, "train"),
        "dans_target": CircularRecordIterator(target_id, "train")
    },
    "test": {
        "source": CircularRecordIterator(source_id, "test"),
        "target": CircularRecordIterator(target_id, "test")
    },
    "validation": {
        "source": CircularRecordIterator(source_id, "val"),
        "target": CircularRecordIterator(target_id, "val")
    }
}

stats = dict()

for mode in iterators:

    if mode not in stats:
        stats[mode] = dict()

    for submode in iterators[mode]:

        nb_positive = 0
        nb_data = 0
        iterator = iterators[mode][submode]

        is_new_epoch = False
        while not is_new_epoch:
            serialized_data, is_new_epoch = iterator.get_next_serialized_batch()
            q, p, y = parse_serialized_batch(serialized_data=serialized_data, batch_size=batch_size)

            nb_data += batch_size
            nb_positive += sum(y)

        stats[mode][submode] = {
            "nb_data": nb_data,
            "nb_positive": nb_positive
        }

        logger.info("Stats for {} {}:".format(mode, submode))
        logger.info(stats[mode][submode])

logger.info("Final stats:\n-----------")
logger.info(stats)
