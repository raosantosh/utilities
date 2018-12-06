import tensorflow as tf
import numpy as np

# Global constants
path_data = '/var/opt/amin/Data/datasets/'
nb_triplets_query_product_buckets = 2 ** 13
batch_positives = 4
negative_samples_factor = 7
batch_size = batch_positives * (negative_samples_factor + 1)  # 32


def get_data_path(catalog_id, mode):
    return path_data + "positive_negative_1p_7n_32_partner{}_{}.proto".format(
        str(catalog_id), mode)


class CircularRecordIterator(object):

    def __init__(self, catalog_id, mode):
        # Saving data path
        self._data_path = get_data_path(catalog_id, mode)
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

    def get_next_batch(self):
        serialized_batch, is_new_epoch = self.get_next_serialized_batch()
        return CircularRecordIterator. \
            parse_serialized_batch(serialized_batch), is_new_epoch

    @staticmethod
    def parse_serialized_batch(serialized_data):
        serialized_batch = tf.train.Example()
        serialized_batch.ParseFromString(serialized_data)

        # Extracting labels
        batch_labels = np.array(
            serialized_batch.features.feature['label'].int64_list.value).reshape(
            batch_size, 1)

        # Extracting queries
        queries_indices = np.array(
            serialized_batch.features.feature['query'].int64_list.value)
        batch_queries = np.zeros((batch_size * nb_triplets_query_product_buckets))
        batch_queries[queries_indices] = 1
        batch_queries = batch_queries.reshape(batch_size,
                                              nb_triplets_query_product_buckets)

        # Extracting products
        products_indices = np.array(
            serialized_batch.features.feature['product'].int64_list.value)
        batch_products = np.zeros((batch_size * nb_triplets_query_product_buckets))
        batch_products[products_indices] = 1
        batch_products = batch_products.reshape(batch_size,
                                                nb_triplets_query_product_buckets)

        return {
            "queries": batch_queries,
            "products": batch_products,
            "labels": batch_labels
        }
