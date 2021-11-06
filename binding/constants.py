"""
Constants used for the binding prediction
"""

AFFINITY_URL = 'https://bmm.crick.ac.uk/~bmmadmin/Affinity/display_table2.cgi?sort=DeltaG&order=asc'
DATA_DIRECTORY = 'data/'
PARATOPE_DIRECTORY = '../paratope/'
AFFINITY_CSV = f"{DATA_DIRECTORY}/affinity_db.csv"
PROBS_CSV = f"{DATA_DIRECTORY}/binding_probabilities.csv"
SABDAB_CSV = f"{PARATOPE_DIRECTORY}/{DATA_DIRECTORY}/sabdab_27_jun_95_90.csv"
SABDAB_TSV = f"{PARATOPE_DIRECTORY}/{DATA_DIRECTORY}/sabdab_summary_all.tsv"