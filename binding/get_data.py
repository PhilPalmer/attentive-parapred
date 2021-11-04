import os.path
import pandas as pd
import sys
import urllib.request

from constants import *

def html_to_csv(download_url, csv_fname):
  """
  Extract tabular data from a HTML webpage and save as CSV
  :param download_url: URL of HTML webpage to download containing tabular data
  :param csv_fname: Output CSV file path
  """
  html = urllib.request.urlopen(download_url).read()
  df = pd.read_html(html)[-1]
  df.to_csv(csv_fname)

def download_pdb(pdb_id, data_dir, download_url="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdb_id: The standard PDB ID e.g. '3ICB' or '3icb'
    :param data_dir: The directory where the downloaded file will be saved
    :param download_url: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdb_fname = pdb_id + ".pdb"
    url = download_url + pdb_fname
    out_fname = os.path.join(data_dir, pdb_fname)
    try:
        urllib.request.urlretrieve(url, out_fname)
        return out_fname
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None

if __name__ == '__main__':
  if not os.path.exists(AFFINITY_CSV):
    html_to_csv(AFFINITY_URL, AFFINITY_CSV)