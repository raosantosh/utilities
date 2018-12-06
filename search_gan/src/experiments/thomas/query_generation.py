class QueryGeneration(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/thomas/query_generation_binary.py"
    GRID = {
        "-learning_rate": [0.00005],
        "-p_keep_for_dropout": [0.8],
        "-catalog_id": ["131"],
        "-method": ["dnn"],
        "-learn_on_negatives": [0, 1]
    }
