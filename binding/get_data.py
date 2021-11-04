import Bio
from Bio.PDB import PDBList
import os
import pandas as pd
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
  df.to_csv(csv_fname)

def download_pdbs(pdb_list, data_dir):
  """
  Download PDB files and save in the specified data directory.
  :param pdb_list: A list of standard PDB IDs to download e.g. `['4B97','4IPH','4HNO']`
  :param data_dir: The directory where the downloaded file will be saved
  """
  pdbl = PDBList()
  for pdb_code in pdb_list:
      pdb_fname = pdbl.retrieve_pdb_file(pdb_code, pdir=data_dir, file_format='pdb')
      os.rename(pdb_fname, f"{data_dir}/{pdb_code}.pdb")

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

if __name__ == '__main__':
  if not os.path.exists(AFFINITY_CSV):
    html_to_csv(AFFINITY_URL, AFFINITY_CSV)