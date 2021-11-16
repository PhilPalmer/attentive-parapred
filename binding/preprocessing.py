import Bio
from Bio.PDB import PDBList
import os
import pandas as pd
import subprocess
import sys
import urllib.request

from constants import *

def html_to_csv(download_url, csv_fname):
  """
  Extract tabular data from a HTML webpage and save as CSV.
  :param download_url: URL of HTML webpage to download containing tabular data
  :param csv_fname: Output CSV file path
  """
  html = urllib.request.urlopen(download_url).read()
  df = pd.read_html(html)[-1]
  df.columns = df.columns.str.replace('<>', '')
  df.to_csv(csv_fname, index=False)

def download_pdbs(pdb_list, data_dir):
  """
  Download PDB files and save in the specified data directory.
  :param pdb_list: A list of standard PDB IDs to download e.g. `['4B97','4IPH','4HNO']`
  :param data_dir: The directory where the downloaded files will be saved
  """
  pdbl = PDBList()
  for pdb_code in pdb_list:
      pdb_fname = pdbl.retrieve_pdb_file(pdb_code, pdir=data_dir, file_format='pdb')
      os.rename(pdb_fname, f"{data_dir}/{pdb_code}.pdb")

def download_affinity_pdbs(affinity_csv, data_dir):
  """
  Download PDB files listed in input Affinity CSV file if they don't already exist in the specified data directory.
  :param affinity_csv: Affinity DB CSV file path containing the column `Complex` for PDB codes
  :param data_dir: The directory where the downloaded file will be saved
  """
  affinity_df = pd.read_csv(affinity_csv)
  pdb_list = affinity_df['Complex'].str[:4].tolist()
  for pdb_code in pdb_list:
    pdb_fname = f"{data_dir}/{pdb_code}.pdb"
    if os.path.exists(pdb_fname):
      pdb_list.remove(pdb_code)
  download_pdbs(pdb_list, data_dir)

def download_data(data_dir=DATA_DIRECTORY, affinity_url=AFFINITY_URL, affinity_csv=AFFINITY_CSV):
  """
  Download Affinity DB and associated PDB files if they don't already exist in the specified data directory.
  :param data_dir: The directory where the downloaded file will be saved
  :param affinity_url: URL of the Affinity DB HTML webpage to download containing tabular data
  :param affinity_csv: Affinity DB CSV file path
  """
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  if not os.path.exists(affinity_csv):
    html_to_csv(affinity_url, affinity_csv)
  download_affinity_pdbs(affinity_csv, data_dir)

def clean_sabdab_data(sabdab_csv=SABDAB_CSV, sabdab_tsv=SABDAB_TSV):
  """
  Get clean list of CSV filtered SAbDab complexes from full SAbDab structural database.
  (This fixes the original CSV file in the repository which was malformed as the commas had not been escaped properly).
  :param sabdab_csv: CSV file path for filtered list of SAbDab complexes of interest to be overwritten
  :param sabdab_tsv: TSV summary file path for full SAbDab structural database
  """
  # Get unique IDs for each PDB file in the SabDab input file
  sabdab_df = pd.read_csv(sabdab_csv)
  sabdab_df['id'] = sabdab_df['pdb'] + sabdab_df['Hchain'] + sabdab_df['Lchain'] + sabdab_df['antigen_chain']
  sabdab_ids = sabdab_df['id'].tolist()
  # Keep only those rows of interest from the full SabDab database input file
  sabdab_df = pd.read_csv(sabdab_tsv, sep='\t')
  sabdab_df['id'] = sabdab_df['pdb'] + sabdab_df['Hchain'] + sabdab_df['Lchain'] + sabdab_df['antigen_chain']
  sabdab_df = sabdab_df[sabdab_df['id'].isin(sabdab_ids)]
  del sabdab_df['id']
  sabdab_df.to_csv(sabdab_csv, index=False)

def get_predictions(paratope_dir=PARATOPE_DIRECTORY, data_directory=DATA_DIRECTORY, sabdab_csv=SABDAB_CSV, probs_csv=PROBS_CSV):
  """
  Get binding probability predictions for complexes with binding strength values
  :param paratope_dir: Path to the directory containing the paratope files
  :param data_dir: The data directory inside the paratope directory
  :param sabdab_csv: CSV file path for filtered list of SAbDab complexes of interest to be overwritten
  :probs_csv: CSV file path for output binding probabilities
  """
paratope_dir=PARATOPE_DIRECTORY
data_directory=DATA_DIRECTORY
sabdab_csv=SABDAB_CSV
probs_csv=PROBS_CSV
  sabdab_df = pd.read_csv(sabdab_csv)
  sabdab_df = sabdab_df[sabdab_df['delta_g'] != 'None']
  pdb_list = sabdab_df['pdb'].tolist() # pdb_list = ['5mi0', '2r56', '1sy6']
  python_path = sys.executable
  probs_dict = {'pdb': [], 'atom': [], 'residue': [], 'paratope_probability': [], 'chain_id': [], 'res_seq_num': []}
  for pdb_code in pdb_list:
    # Get predictions
    bash_cmd = f"cd {paratope_dir}; {python_path} library_commands.py pdb {pdb_code} --model AFPX"
    subprocess.run(bash_cmd, shell=True)
    # Save predictions
    pdb_fname = f"{paratope_dir}/{data_directory}/{pdb_code}.pdb"
    atom = 0
    with open(pdb_fname) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
      if line.startswith('ATOM') or line.startswith('HETATM'):
        atom += 1
        probs_dict['pdb'].append(pdb_code)
        probs_dict['atom'].append(atom)
        probs_dict['residue'].append(line[16:20])
        probs_dict['paratope_probability'].append(line[60:66])
        probs_dict['chain_id'].append(line[21])
        probs_dict['res_seq_num'].append(line[22:26])
  probs_df = pd.DataFrame(probs_dict)
  # Filter by binding probability
  # probs_df[probs_df['paratope_probability'].astype(float) > 0.95]
  probs_df.to_csv(probs_csv, index=False)

if __name__ == '__main__':
  # download_data()
  # clean_sabdab_data()
  get_predictions()