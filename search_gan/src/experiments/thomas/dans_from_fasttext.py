class DansFromFastText(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/thomas/dans_from_fasttext_binary.py"
    GRID = {
        "-learning_rate": [0.005],
        "-learning_rate_dans": [0.0005],
        "-p_keep_for_dropout": [0.5],
        "-noise_level_for_domain_labels": [0.0],
        "-embedding_size": [25],
        "-dssm_layer_1_size": [50],
        "-dssm_layer_2_size": [50],
        "-discriminator_layer_size": ["100"],
        "-activate_dan_flow": [0, 1],
        "-repeat": [1],
        "-source_id_and_target_id": ["164_180"],
        "-max_step_for_training": [100000],
        "-max_step_for_validation": [4000],
        "-independent_query_embedding": [0],
        "-mode": ["dann_product"],
        "-input_mode": ["new"]
    }
