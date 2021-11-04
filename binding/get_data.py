import urllib
import os.path
import pandas as pd

from constants import *

def html_to_csv(url, csv):
  """
  Extract tabular data from a HTML webpage and save as CSV
  :param url: URL of HTML webpage containing tabular data
  :param csv: Output CSV file path
  """
  html = urllib.request.urlopen(url).read()
  df = pd.read_html(html)[-1]
  df.to_csv(csv)

if __name__ == '__main__':
  if not os.path.exists(AFFINITY_CSV):
    html_to_csv(AFFINITY_URL, AFFINITY_CSV)