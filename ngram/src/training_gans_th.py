import tensorflow as tf
import os
import shutil
import numpy as np
from tqdm import tqdm
from training.tensorflow.simple_model import get_logit_model, get_model_path
from datasets.taxonomy import TaxonomyHelper
from datasets.batcher import VectorBatchLabeledDataset
from datasets.toy_rotater import U
from utils import get_run_id

# Constant tensors
u_tensor = tf.constant(U, dtype=tf.float32)
i_tensor = tf.constant(np.identity(300), dtype=tf.float32)

# Constant values
batch_size = 500
batch_size_alt = 500

print("Sanity check: {}".format(np.linalg.norm(np.matmul(U, np.transpose(U))-np.identity(300))))


def get_registration_model(x):
    """
    Create the model for the registration
    :param x:
    :return:
    """
    with tf.variable_scope('T_register'):
        t = tf.get_variable("t", shape=[300, 300], initializer=tf.orthogonal_initializer)
    # Applying the transformation
    x_registered = tf.matmul(x, t)
    return x_registered, t


def get_complex_registration_model(x):
    """
        Create the model for the registration
        :param x:
        :return:
        """
    with tf.variable_scope('T_register'):
        w_reg = tf.get_variable("w_reg", shape=[300, 300], initializer=tf.truncated_normal_initializer)
        b_reg = tf.get_variable("b_reg", shape=[300], initializer=tf.zeros_initializer)
        w_reg_out = tf.get_variable("w_reg_out", shape=[300, 300], initializer=tf.truncated_normal_initializer)
        b_reg_out = tf.get_variable("b_reg_out", shape=[300], initializer=tf.zeros_initializer)
    # Applying the transformation
    x_layer_1 = tf.nn.relu(tf.matmul(x, w_reg) + b_reg)
    x_layer_2 = tf.nn.tanh(tf.matmul(x_layer_1, w_reg_out) + b_reg_out)
    return x_layer_2


def orthogonalize_t():
    """
    Force the tensor T to be orthogonal b running a QR decomposition
    :return:
    """
    t = tf.get_default_graph().get_tensor_by_name('T_register/t:0')
    q, _ = tf.qr(t)
    return tf.assign(t, q)


def get_discriminant_model(x):
    """
    Create the model for the discriminator
    :param x:
    :param x_registered:
    :return:
    """
    with tf.variable_scope('T_discriminator', reuse=tf.AUTO_REUSE):
        w_discr = tf.get_variable("w_discr", shape=[300, 2], initializer=tf.truncated_normal_initializer)
        b_discr = tf.get_variable("b_discr", shape=[2], initializer=tf.zeros_initializer)
    y_discr_logit = tf.matmul(x, w_discr) + b_discr
    return y_discr_logit


def do_learn():
    # Gradient step placeholder
    counterfeiter_step_size = tf.placeholder(tf.float32, name="counterfeiter_step_size")
    counterfeiter_regularization_lambda = tf.placeholder(tf.float32, name="counterfeiter_regularization_lambda")

    # Input vectors (300d pretrained fastText vector)
    x = tf.placeholder(tf.float32, [None, 300], name="x")
    x_alt = tf.placeholder(tf.float32, [None, 300], name="x_alt")

    x_summary = tf.summary.histogram("x", x)
    x_alt_summary = tf.summary.histogram("x_alt", x_alt)

    # Input label (first level ; only 21 classes)
    # Will be used to measure classifier performance
    y_label_category = tf.placeholder(tf.int32, [None], name="y_label_category")
    y_label_category_one_hot = tf.one_hot(y_label_category, 21, name="y_label_category_one_hot")

    y_label_discr = tf.placeholder(tf.int32, [None], name="y_label_discriminator")
    y_label_discr_one_hot = tf.one_hot(y_label_discr, 2, name="y_label_discriminator_one_hot")

    y_label_category_alt = tf.placeholder(tf.int32, [None], name="y_label_category_alt")
    y_label_category_alt_one_hot = tf.one_hot(y_label_category_alt, 21, name="y_label_category_alt_one_hot")

    y_label_discr_alt = tf.placeholder(tf.int32, [None], name="y_label_discriminator_alt")
    y_label_discr_one_hot_alt = tf.one_hot(y_label_discr_alt, 2, name="y_label_discriminator_alt_one_hot")

    # Creating model for registration (simple matrix multiplication)
    # Note: we apply the transformation only on alt inputs
    # x_alt_registered, t = get_registration_model(x_alt)
    x_alt_registered, t = get_registration_model(x_alt)
    x_alt_registered_summary = tf.summary.histogram("Deep registration activation", x_alt_registered)
    # Adding op to compute the norm of t
    # Adding op to compute the norm of t-U (useless for toy rotated experiment)
    t_minus_u_norm = tf.norm(tf.add(t, -u_tensor))
    ttt_minus_id_norm = tf.norm(tf.matmul(t, tf.transpose(t)) - i_tensor)

    # Adding summaries
    ttt_minus_id_norm_summary = tf.summary.scalar("Norm of T*T^T-Id", ttt_minus_id_norm)
    t_minus_u_norm_summary = tf.summary.scalar("Norm of T-U", t_minus_u_norm)
    # t_histogram = tf.summary.histogram("Values in T", t)

    # Creating discriminating model
    y_discr_logit = get_discriminant_model(x)
    y_discr_logit_alt = get_discriminant_model(x_alt_registered)

    # Labels for discriminator (2 classes)
    # Label = 0 <=> is real sample
    # Label = 1 <=> is fake sample

    # Defining loss for discriminator
    real_discr_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_label_discr_one_hot,
        logits=y_discr_logit))

    real_discr_loss_summary = tf.summary.scalar("Discriminator real samples loss",
                                                real_discr_loss/batch_size)

    fake_discr_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_label_discr_one_hot_alt,
        logits=y_discr_logit_alt))

    fake_discr_loss_summary = tf.summary.scalar("Discriminator fake samples loss",
                                                fake_discr_loss/batch_size_alt)

    discriminator_loss = real_discr_loss + fake_discr_loss

    discriminator_loss_summary = tf.summary.scalar("Discriminator total loss",
                                                   discriminator_loss/(batch_size+batch_size_alt))

    # Defining loss for counterfeiter
    counterfeiter_regularization = counterfeiter_regularization_lambda*tf.norm(t)

    counterfeiter_regularization_summary = tf.summary.scalar("Counterfeiter regularization",
                                                             counterfeiter_regularization/batch_size_alt)

    counterfeiter_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=1-y_label_discr_one_hot_alt,
        logits=y_discr_logit_alt)) + tf.norm(x_alt-x_alt_registered)

    counterfeiter_loss_summary = tf.summary.scalar("Counterfeiter loss",
                                                   counterfeiter_loss/batch_size_alt)
    counterfeiter_loss_w_reg_summary =\
        tf.summary.scalar("Counterfeiter loss w regul",
                          (counterfeiter_loss + counterfeiter_regularization)/batch_size_alt)

    # Loading model for the classifier part
    y_model_logits = get_logit_model(x)
    y_model_logits_alt = get_logit_model(x_alt_registered)

    # Metric for the classifier
    is_correct = tf.equal(tf.argmax(y_model_logits, 1), tf.argmax(y_label_category_one_hot, 1))
    accuracy_classifier = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # Metric for the classifier alt
    is_correct_alt = tf.equal(tf.argmax(y_model_logits_alt, 1), tf.argmax(y_label_category_alt_one_hot, 1))
    accuracy_classifier_alt = tf.reduce_mean(tf.cast(is_correct_alt, tf.float32))

    accuracy_classifier_summary = tf.summary.scalar("Classifier accuracy on real data", accuracy_classifier)
    accuracy_classifier_alt_summary = tf.summary.scalar("Classifier accuracy on fake data", accuracy_classifier_alt)

    # In this experiment, we want to learn only the registration so we fix the variables to be trained
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "T_discriminator")
    print("Discriminator can update the following variables: {}".format(discriminator_variables))
    counterfeiter_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "T_register")
    print("Counterfeiter can update the following variables: {}".format(counterfeiter_variables))
    classifier_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "f_classifier")

    discriminator_optimizer = tf.train.AdamOptimizer(1e-3)
    counterfeiter_optimizer = tf.train.AdamOptimizer(counterfeiter_step_size)

    discriminator_step = discriminator_optimizer.minimize(discriminator_loss, var_list=discriminator_variables)
    counterfeiter_step =\
        counterfeiter_optimizer.minimize(counterfeiter_loss + counterfeiter_regularization,
                                         var_list=counterfeiter_variables)

    # Datasets used in this experiment

    train_dataset = os.path.join(os.path.dirname(__file__),
                                 "../../resources/processed_data/EN_full_labeled_dpe_pop_0_vectors.txt")
    train_labels = os.path.join(os.path.dirname(__file__),
                                "../../resources/processed_data/EN_full_labeled_dpe_pop_0_labels.txt")

    train_batcher = VectorBatchLabeledDataset(data_path=train_dataset, label_path=train_labels)

    test_dataset = os.path.join(os.path.dirname(__file__),
                                "../../resources/processed_data/EN_full_labeled_dpe_pop_2_vectors.txt")
    test_labels = os.path.join(os.path.dirname(__file__),
                               "../../resources/processed_data/EN_full_labeled_dpe_pop_2_labels.txt")

    test_batcher = VectorBatchLabeledDataset(data_path=test_dataset, label_path=test_labels)

    train_dataset_fr = os.path.join(os.path.dirname(__file__),
                                    "../../resources/processed_data/EN_full_labeled_dpe_pop_0_vectors_rotated.txt")
    train_labels_fr = os.path.join(os.path.dirname(__file__),
                                   "../../resources/processed_data/EN_full_labeled_dpe_pop_0_labels.txt")

    train_batcher_fr = VectorBatchLabeledDataset(data_path=train_dataset_fr, label_path=train_labels_fr)

    test_dataset_fr = os.path.join(os.path.dirname(__file__),
                                   "../../resources/processed_data/EN_full_labeled_dpe_pop_2_vectors_rotated.txt")
    test_labels_fr = os.path.join(os.path.dirname(__file__),
                                  "../../resources/processed_data/EN_full_labeled_dpe_pop_2_labels.txt")

    test_batcher_fr = VectorBatchLabeledDataset(data_path=test_dataset_fr, label_path=test_labels_fr)

    # Selecting 100k test points
    x_test, y_test = test_batcher.nextBatch(100000)
    y_test_first_level = np.array([TH.get_first_level_index_for_id(cat_id) for cat_id in y_test])

    # Selecting 100k test points from french dataset
    x_test_fr, y_test_fr = test_batcher_fr.nextBatch(100000)
    y_test_first_level_fr = np.array([TH.get_first_level_index_for_id(cat_id) for cat_id in y_test_fr])

    # Defining initializer / session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Reloading pre-computed model (only for classifier)
    print("Loading fixed classifier model...")
    saver = tf.train.Saver(classifier_variables)
    saver.restore(sess, get_model_path())
    print("Classifier model successfully loaded !")

    def get_losses():
        loss, floss, rloss = sess.run([discriminator_loss, fake_discr_loss, real_discr_loss],
                                      feed_dict=train_data)
        loss = loss/(batch_size+batch_size_alt)
        floss = floss/batch_size_alt
        rloss = rloss/batch_size
        print("Real sample loss: {}".format(rloss))
        print("Fake sample loss: {}".format(floss))
        print("Discriminator loss: {}".format(loss))
        return loss, floss, rloss

    ##################################
    # Defining tensorboard summaries #
    ##################################

    train_summaries = tf.summary.merge([
        ttt_minus_id_norm_summary,
        t_minus_u_norm_summary,
        real_discr_loss_summary,
        fake_discr_loss_summary,
        discriminator_loss_summary,
        counterfeiter_loss_summary,
        counterfeiter_loss_w_reg_summary,
        #t_histogram,
        x_alt_registered_summary,
        x_summary,
        x_alt_summary,
        counterfeiter_regularization_summary
    ], "Train summaries")

    test_summaries = tf.summary.merge([
        accuracy_classifier_summary,
        accuracy_classifier_alt_summary
    ], "Test summaries")

    log_location = '/home/ubuntu/ucat/logs/{}'.format(get_run_id())
    print("Using log location for tensorboard {}".format(log_location))
    shutil.rmtree(log_location, ignore_errors=True)
    os.mkdir(log_location)

    summary_writer = tf.summary.FileWriter(log_location,
                                           sess.graph)

    def get_counterfeiter_step_size(it):
        return 1e-3  # np.max([10*np.exp(-0.5*it), 1e-1])

    def get_counterfeiter_regularization_lambda(it):
        return 100
        #if it < 250:
        #    return 100
        #elif it < 500:
        #    return 20
        #elif it < 1000:
        #    return 10
        #else:
        #    return 5

    #######################
    # Starting iterations #
    #######################

    hysteresis_bounds = [0.9 * np.log(2), 1.1 * np.log(2)]
    hysteresis_counter = 0

    print("Starting iteration")
    for it in tqdm(range(2000)):

        x_train_en, y_train_en = train_batcher.nextBatch(batch_size)
        x_train_fr, y_train_fr = train_batcher_fr.nextBatch(batch_size_alt)

        # Updating discriminator and counterfeiter
        train_data = {
            x: x_train_en,
            x_alt: x_train_fr,
            y_label_discr: np.zeros(batch_size),
            y_label_discr_alt: np.ones(batch_size_alt),
            counterfeiter_step_size: get_counterfeiter_step_size(it),
            counterfeiter_regularization_lambda: get_counterfeiter_regularization_lambda(it)
        }

        def run_discriminator():
            #print("Updating discriminator...")
            sess.run([discriminator_step], feed_dict=train_data)

        def run_counterfeiter(it):
            #print("Updating counterfeiter...")
            sess.run([counterfeiter_step], feed_dict=train_data)
            #print("Orthogonalizing T")
            #sess.run([orthogonalize_t()])

        current_discriminator_loss = sess.run([discriminator_loss], feed_dict=train_data)[0]/(batch_size+batch_size_alt)
        # If the discriminator is worse than random, let's improve it first
        if current_discriminator_loss > hysteresis_bounds[hysteresis_counter]:
            run_discriminator()
            hysteresis_counter = 1 - hysteresis_counter
        # Else let's improve the counterfeiter 2/3 of the time
        elif int(it/10) % 3 == 2:
            run_discriminator()
        else:
            run_counterfeiter(it)

        # Sending summaries to tensorboard
        train_sum = sess.run([train_summaries], feed_dict=train_data)[0]
        summary_writer.add_summary(train_sum, it)

        if it % 10 == 0:
            # Sending test summaries (accuracy of f on english data should not vary)
            test_data = {
                x: x_test,
                x_alt: x_test_fr,
                y_label_category: y_test_first_level,
                y_label_category_alt: y_test_first_level_fr
            }

            # Sending test summaries
            test_sum = sess.run([test_summaries], feed_dict=test_data)[0]
            summary_writer.add_summary(test_sum, it)


if __name__ == "__main__":
    print("Using tensorflow version {}".format(tf.__version__))
    print("Using release {}".format(get_run_id()))
    do_learn()
