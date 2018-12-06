import tensorflow as tf
import functools
import numpy as np
import typing
import os
import argparse
import logging
logging.basicConfig(level="INFO")

# Parameters from grid
parser = argparse.ArgumentParser(
    description='Prepare datasets for a given catalog')
parser.add_argument('--catalog_id', type=int)

args = parser.parse_args()

root_path = "/var/opt/thomas/splitted_embeddings"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def parse_row(row):
    splitted = tf.sparse_tensor_to_dense(tf.string_split([row], " "), default_value="")
    return tf.string_to_number(splitted)


def merge_product_and_query_ds(products, queries, label):
    label_tensor = tf.constant([label], dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices({
        "product": products,
        "query": queries,
        "label": label_tensor
    })


def flatten_interleave_ds(*args):
    products = tf.concat([tf.expand_dims(arg["product"], axis=0) for arg in args], axis=0)
    queries = tf.concat([tf.expand_dims(arg["query"], axis=0) for arg in args], axis=0)
    labels = tf.stack([arg["label"] for arg in args], axis=0)
    return tf.data.Dataset.from_tensor_slices({
        "product": products,
        "query": queries,
        "labels": tf.reshape(labels, shape=(-1, 1))
    })


datasets = dict()

for split in ["train", "test"]:
    dataset_products = tf.data.TextLineDataset(
            filenames=[f"{root_path}/{split}_productnames_{args.catalog_id}_embeddings.csv"],
            buffer_size=int(1e6)).map(parse_row, num_parallel_calls=4)

    dataset_queries = tf.data.TextLineDataset(
            filenames=[f"{root_path}/{split}_queries_{args.catalog_id}_embeddings.csv"],
            buffer_size=int(1e6)).map(parse_row, num_parallel_calls=4)

    dataset_products_negative = tf.data.TextLineDataset(
            filenames=[f"{root_path}/{split}_productnames_{args.catalog_id}_negative_embeddings.csv"],
            buffer_size=int(1e6)).map(parse_row, num_parallel_calls=4)

    nb_negative_for_one_positive = 7
    batch_size = 20 * (nb_negative_for_one_positive + 1)

    # Zipping queries and products for positive samples
    ds_pos = tf.data.Dataset.zip((
        dataset_products,
        dataset_queries))
    ds_pos = ds_pos.flat_map(functools.partial(merge_product_and_query_ds, label=1))

    # Zipping queries and products for negative samples
    ds_neg = tf.data.Dataset.zip((
        dataset_products_negative.shuffle(512, seed=37),
        dataset_queries))
    ds_neg = ds_neg.flat_map(functools.partial(merge_product_and_query_ds, label=0))

    # Interleaving positive and negative
    ds_merged = [ds_pos]
    ds_merged.extend(
        [ds_neg.shuffle(512, seed=37 * (i + 1))
         for i in range(nb_negative_for_one_positive)]
    )
    ds_zip = tf.data.Dataset.zip(tuple(ds_merged))

    ds_final = ds_zip.flat_map(flatten_interleave_ds)

    datasets[split] = ds_final


def _int64_feature(values: typing.List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values: typing.List[str]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode() for v in values]))


def _float_feature(values: typing.List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


session = tf.Session()

for split in ["train", "test"]:
        ds = datasets[split]
        writer = tf.python_io.TFRecordWriter(f"{root_path}/ds_{split}_{args.catalog_id}")
        next_op = ds.make_one_shot_iterator().get_next()

        while True:
            try:
                elem = session.run(next_op)

                feature = {
                    "product": _float_feature(elem["product"]),
                    "query": _float_feature(elem["query"]),
                    "labels": _int64_feature([elem["labels"]])
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                writer.write(example.SerializeToString())

            except tf.errors.OutOfRangeError:
                print(f"End of validation dataset for {split}.")
                writer.close()
                break

print("Done")