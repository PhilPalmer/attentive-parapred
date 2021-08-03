# Code for [Attentive cross-modal paratope prediction](https://arxiv.org/pdf/1806.04398)

*Note: I don't think these instructions will run the package exactly as intended. This is just how I got it to run*

## Install
```bash
# Download repository
git clone https://github.com/PhilPalmer/attentive-parapred.git && cd attentive-parapred

# Optional - setup conda environment
conda create --name parapred -y
conda activate parapred

# Install dependencies
python setup.py install
```

## Usage
```bash
cd paratope # Note you must be in the paratope directory so that `data/sabdab_27_jun_95_90.csv` can be found
python library_commands.py --help # Display the help message below
python library_commands.py pdb 4bz1 --model FP # Run the Fast-Parapred model on the 4bz1 complex
```

Help output
```
Paratope prediction.

Usage:
    fast_parapred cdr <cdr_seq> [--chain <chain>]
    fast_parapred pdb <pdb_name> [--model <predictor>] [--abh <ab_h_chain>] [--abl <ab_l_chain>] [--ag <ag_chain>]
    fast_parapred --help

Options:
    cdr <cdr_seq>               The input should be a CDR sequence with 2 additional residues at either end.
                                The outputs will consist of a binding probability for each amino acid.
    --chain <chain>             The name of the chain. It has to be one of {H1, H2, H3, L1, L2, L4}.
    pdb <pdb_name>              Given a PDB file name and the names of the high and the low chain,
                                it replaces the temperature factor with
                                binding probabilities.
    --abh <ab_h_chain>       Name of the antibody high chain.
    --abl <ab_l_chain>       Name of the antibody low chain.
    --model <m>                 Predictor to be used: LSTM Baseline(L), Parapred(P), Fast-Parapred(FP) or
                                    AG-Fast-Parapred (AFP).
    --ag <ag_chain>          Name of antigen chain in PDB file.
    -h --help                    Show this help.
```