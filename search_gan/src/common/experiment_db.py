import sqlite3
from contextlib import contextmanager
import datetime
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ExperimentDB]")


@contextmanager
def db_cursor():
    conn = sqlite3.connect('experiment.db')
    cursor = conn.cursor()
    try:
        yield cursor
    except Exception as e:
        logger.error("Error: {}".format(e))
        conn.rollback()
    else:
        conn.commit()


# Initialize experiments table
def init_experiment_table():
    with db_cursor() as cur:

        cur.execute('''CREATE TABLE IF NOT EXISTS experiments
                     (
                     metarun_id int,
                     run_id int,
                     experiment_name text,
                     date text,
                     parameters text,
                     metrics text,
                     PRIMARY KEY (metarun_id, run_id)
                     )''')


def get_nb_experiments():
    with db_cursor() as cur:
        cur.execute("SELECT count(1) FROM experiments")
        nb_experiments = cur.fetchone()[0]

    return nb_experiments


def add_experiment(metarun_id, run_id, experiment_name, parameters):
    with db_cursor() as cur:

        date = str(datetime.datetime.now().isoformat())
        print(date)

        cur.execute("""INSERT INTO experiments VALUES (
            {metarun_id},
            {run_id},
            '{experiment_name}',
            '{date}',
            '{parameters}',
            ''
        )
        """.format(
            metarun_id=metarun_id,
            run_id=run_id,
            experiment_name=experiment_name,
            date=date,
            parameters=json.dumps(parameters)
        ))


def update_experiment(metarun_id, run_id, metrics):
    with db_cursor() as cur:
        cur.execute("""UPDATE experiments 
        SET metrics = '{metrics}' 
        WHERE metarun_id = {metarun_id} AND run_id = {run_id}""".format(
            metrics=json.dumps(metrics),
            run_id=run_id,
            metarun_id=metarun_id
        ))


def list_all_experiments():
    with db_cursor() as cur:
        ret = list()
        for row in cur.execute("SELECT * FROM experiments"):
            ret.append(row)

    return pd.DataFrame(ret, columns=["metarun_id", "run_id", "experiment_name", "date", "parameters", "metrics"])


if __name__ == "__main__":
    init_experiment_table()
    print("Found {} experiments".format(get_nb_experiments()))
    add_experiment(metarun_id=1, run_id=2, experiment_name="test", parameters={"a": 37})
    final_metrics = {'source': {'xentropy': 0.12906879332720064, 'auc': 0.53197604}, 'target': {'xentropy': 0.12289614128131493, 'auc': 0.6172068}, 'source_dans': {'xentropy': 0.13211117567969302, 'auc': 0.5320687}}
    update_experiment(metarun_id=1, run_id=2, metrics=final_metrics)
    print(list_all_experiments().iloc[0])
