class DansFromProto(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/thomas/dans_from_proto_binary.py"
    GRID = {
        "-learning_rate": [0.00005],
        "-learning_rate_dans": [0.00005],
        "-p_keep_for_dropout": [1.0],
        "-noise_level_for_domain_labels": [0.0],
        "-embedding_size": [50],
        "-dssm_layer_1_size": [300],
        "-dssm_layer_2_size": [100],
        "-discriminator_layer_size": [50],
        "-activate_dan_flow": [1],
        "-repeat": [1, 2],
        "-source_id_and_target_id": ["193_101"],
        "-max_epoch_for_training": [1],
        "-independent_query_embedding": [0]
    }
