class DansFromTriplets(object):
    MAX_NB_PROCESSES = 3
    DEBUG = False
    BINARY = "experiments/thomas/dans_from_triplets_binary.py"
    GRID = {
        "-learning_rate": [0.0005],
        "-learning_rate_dans": [0.0005],
        "-p_keep_for_dropout": [0.5],
        "-noise_level_for_domain_labels": [0.0],
        "-embedding_size": [25],
        "-dssm_units": ["25_25"],
        "-discriminator_units": ["100"],
        "-activate_dan_flow": [1],
        "-repeat": [1],
        "-source_id_and_target_id": ["193_101"],
        "-max_step_for_training": [100000],
        "-max_step_for_validation": [4000],
        "-independent_query_embedding": [0, 1],
        "-mode": ["source", "dann_product"]
    }
