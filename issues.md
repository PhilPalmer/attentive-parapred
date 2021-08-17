# General

I encountered a number of issues trying to run the code. I have made a note of and characterised the issues here

:heavy_check_mark: = issue now resolved

1) Was this code used for [Attentive cross-modal paratope prediction](https://arxiv.org/pdf/1806.04398) then improved upon for [Neural message passing for joint paratope-epitope prediction](https://arxiv.org/pdf/2106.00757.pdf)?

2) Little documentation for how to run the code? (See [#2](https://github.com/andreeadeac22/attentive-parapred/issues/2))

3) What are the inputs? Do both the antibody and antigen need to be input as a complex or can they be seperate PDB files?

<details>
<summary>4. Which model to use? Can the existing models/weights be provided? </summary>

Code for the following models exists. However, it is not clear which model is best to use and the trained weights for each of the models are not included within the repository. Using the code the models can be retrained but it's not clear what run or fold is best to use from the 100 generated files.

| Paper | Model name                            | Code | Description                                                                                     |
|-------|---------------------------------------|------|-------------------------------------------------------------------------------------------------|
| 2018  | LSTM Baseline aka RNNModel            | 6    | Antibody-only paratope prediction, baseline                                                     |
| 2018  | Parapred  aka AbSeqModel              | 1    | Antibody-only paratope prediction, existing model by Liberis et al.                             |
| 2018  | Fast-Parapred aka AtrousSelf          | 5    | Antibody-only paratope prediction, while requiring only half the computational time of Parapred |
| 2018  | AG-Fast-Parapred aka AG               | 4    | Antibody and antigen paratope prediction                                                        |
| 2021? | Cross-self AG-Fast-Parapred aka XSelf | 7    | Improved antibody and antigen paratope prediction?                                              |
| 2021? | AttentionRNN                          | 2    | ?                                                                                               |
| 2021? | DilatedConv                           | 3    | ?                                                                                               |

</details>

# Paratope prediction

I encountered a number of issues with installing and running the paratope prediction. It's possibly that many of these issues are caused by incorrect installation. Some documentation regarding installation would help solve the problem

## Installation

<details>
<summary>5. :heavy_check_mark: `error: package directory 'aid25' does not exist` </summary>

Steps to replicate:
```bash
# Download repo
git clone https://github.com/andreeadeac22/attentive-parapred.git && cd attentive-parapred

# Create conda env
conda create --name parapred_test -y
conda activate parapred_test

# Install dependencies
python setup.py install
```

Fix: Changed package name (see [1ccecc6](https://github.com/PhilPalmer/attentive-parapred/commit/1ccecc618e4cd8fb7fb409a57b3efeb75ffddcf8))

</details>

<details>
<summary>6. :heavy_check_mark: Coudln't install packages such as sci-kit learn </summary>

Error:

Possibly an issue with my development environment and not the package:
```
error: Setup script exited with error: Command "g++ -pthread -B /home/pp502/.conda/envs/parapred/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/numpy/core/include -I/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/numpy/core/include -I/home/pp502/.conda/envs/parapred/include/python3.8 -c sklearn/cluster/_dbscan_inner.cpp -o build/temp.linux-x86_64-3.8/sklearn/cluster/_dbscan_inner.o -MMD -MF build/temp.linux-x86_64-3.8/sklearn/cluster/_dbscan_inner.o.d" failed with exit status 1
```

Fix: Manually installed troublesome packages with conda and commented the installs out from my local version of the `setup.py` script

</details>

## Training (running `main.py`)

To rerun the cross-validation and generate the models

<details>
<summary>7. :heavy_check_mark: Failed to run cross validation for all complexes</summary>

Steps to replicate:
```bash
python main.py
```

Error:

Failed to run cross validation for all complexes in [`paratope/data/sabdab_27_jun_95_90.csv`](paratope/data/sabdab_27_jun_95_90.csv). It took a long while to process and samples and threw an error because a tensor was unexpectantly empty:
```
Computing and storing the dataset...
in load_chains
name A
Processing PDB  4bz1
all_max tensor(54.)
name D
Processing PDB  3gbm
name A
Processing PDB  2qqn
all_max tensor(66.)
name A
Processing PDB  5mes
name A
Processing PDB  2ypv
all_max tensor(110.)
name C
Processing PDB  4uu9
name B

....

Processing PDB  1w72
name 
Processing PDB  3gjf
name B
Processing PDB  3wkm
total dist torch.Size([1419, 32, 1269])
Crossvalidation run 1
cdrs torch.Size([1419, 32, 34])
ag torch.Size([1419, 1269, 28])
Fold:  1
len(train_idx 1277
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
~/Documents/GitHub/attentive-parapred/paratope/main.py in <module>
    147     initial_compute_classifier_metrics(labels, probs, threshold=0.4913739)
    148 
--> 149 run_cv()
    150 #process_cv_results()

~/Documents/GitHub/attentive-parapred/paratope/main.py in run_cv(dataset, output_folder, num_iters)
    103                            str(i) + "-fold-{}.pth.tar"
    104         kfold_cv_eval(dataset,
--> 105                       output_file, weights_template, seed=i)
    106 
    107 def process_cv_results():

~/Documents/GitHub/attentive-parapred/paratope/evaluation.py in kfold_cv_eval(dataset, output_file, weights_template, seed)
    121                             ag_train, ag_masks_train, ag_lengths_train, dist_mat_train, weights_template, i,
    122                             cdrs_test, lbls_test, mask_test, lengths_test,
--> 123                             ag_test, ag_masks_test, ag_lengths_test, dist_mat_test)
    124 
    125         print("test", file=track_f)

~/Documents/GitHub/attentive-parapred/paratope/xself_run.py in xself_run(cdrs_train, lbls_train, masks_train, lengths_train, ag_train, ag_masks_train, ag_lengths_train, dist_mat_train, weights_template, weights_template_number, cdrs_test, lbls_test, masks_test, lengths_test, ag_test, ag_masks_test, ag_lengths_test, dist_test)
    153             #print("Total time", total_time)
    154 
--> 155         print("Epoch %d - loss is %f : " % (epoch, epoch_loss.data[0]/batches_done))
    156         #print("--- %s seconds ---" % (total_time))
    157         times.append(total_time)

IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
```

Fix: As the error is caused by a print statement it can simply be commented out and the script will then run correctly (see [1c7b82d](https://github.com/PhilPalmer/attentive-parapred/commit/1c7b82daac39ed8084ad174e0015cac7e7157206))

</details>

<details>
<summary>8. :heavy_check_mark: Failed to do full run for subset of complexes</summary>

Steps to replicate:
```bash
# Keep only top 10 complexes and make back-up of all complexes
cd paratope/data
cp sabdab_27_jun_95_90.csv sabdab_27_jun_95_90.csv.bak && head -n 10 sabdab_27_jun_95_90.csv.bak > sabdab_27_jun_95_90.csv

# Modify the last 3 lines to do `full_run()` not just cross validation
head -n -3 main.py > main_full_run.py
echo 'full_run()' >> main_full_run.py

# Do full run
python main_full_run.py
```

Error:

For some reason it appears that too few arguments are passed to the function that performs forward propagation. This may be the result of PyTorch updates (see [here](https://discuss.pytorch.org/t/typeerror-forward-missing-2-required-positional-arguments-cap-lens-and-hidden/20010/2)), however, I am not sure because I am using the recommend version in the `requirements.txt`
```
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
~/Documents/GitHub/attentive-parapred/paratope/main.py in <module>
    149 # run_cv()
    150 #process_cv_results()
--> 151 full_run()

~/Documents/GitHub/attentive-parapred/paratope/main.py in full_run(dataset, out_weights)
     65             print("input shape", input.data.shape)
     66             #print("lengths", lengths[j:j+32])
---> 67             output = model(input, lengths[j:j+32])
     68             lbls = index_select(total_lbls, 0, interval)
     69             print("lbls before pack", lbls.shape)

~/anaconda3/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

TypeError: forward() missing 2 required positional arguments: 'masks' and 'lengths'
```

Fix: I am unsure of the exact cause of this error but no fix is required as only the cross-validation can be run instead of the full run and no error is produced

</details>

## Inference (running `fast_parapred`)

<details>
<summary>9. :heavy_check_mark: `ModuleNotFoundError: No module named 'constants'` </summary>

Steps to replicate:
```bash
fast_parapred --help
```

Error:

Could not import other scripts when running the `fast_parapred` command:
```
Traceback (most recent call last):
  File "/home/pp502/.conda/envs/parapred/bin/fast_parapred", line 33, in <module>
    sys.exit(load_entry_point('Fast-Parapred==1.0', 'console_scripts', 'fast_parapred')())
  File "/home/pp502/.conda/envs/parapred/bin/fast_parapred", line 25, in importlib_load_entry_point
    return next(matches).load()
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/importlib/metadata.py", line 77, in load
    module = import_module(match.group('module'))
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/paratope/library_commands.py", line 38, in <module>
    from constants import *
ModuleNotFoundError: No module named 'constants'
```

Fix: Run the `library_commands.py` script instead eg
```
python library_commands.py --help
```

</details>


<details>
<summary>10. :heavy_check_mark: `No such file or directory: 'data/sabdab_27_jun_95_90.csv'` </summary>

Steps to replicate:
```bash
python paratope/library_commands.py --help
```

Error: The SAbDab CSV file cannot be found

Fix: Run the `library_commands.py` script is run in the `paratope` directory so that the SAbDab CSV file can be found. It may also be possible to change the location of this file or make it a parameter in the `constants.py` file or make it an input parameter, however, this is slightly more difficult and I couldn't get it to work instantly

</details>


<details>
<summary>11. :heavy_check_mark: `Attempted relative import beyond top-level package` </summary>

Steps to replicate:
```bash
python library_commands.py --help
```

Error:

This may be because of the way I am trying to run the package but I was getting this error:
```
File "library_commands.py", line 38, in <module>
  from .constants import *
ImportError: attempted relative import with no known parent package
```

Fix: Removed the `.` before the imported filenames in `library_commands.py` (see [38c97b9](https://github.com/PhilPalmer/attentive-parapred/commit/38c97b9da1ea25a05c6d4ad6ad5205f7148bb14f))

</details>


<details>
<summary>12. :heavy_check_mark: Model files cannot be found</summary>

Steps to replicate:
```bash
python library_commands.py pdb 4bz1 --model FP
```

Error:

The model files cannot be found for any of the models (LSTM Baseline(L), Parapred(P), Fast-Parapred(FP) or AG-Fast-Parapred (AFP)):
```
Traceback (most recent call last):
  File "library_commands.py", line 187, in <module>
    main()
  File "library_commands.py", line 179, in main
    process_single_pdb(arguments["<pdb_name>"], arguments["--model"],
  File "library_commands.py", line 163, in process_single_pdb
    model = get_predictor(model_type)
  File "library_commands.py", line 72, in get_predictor
    _model.load_state_dict(torch.load(weights))
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/torch/serialization.py", line 581, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/torch/serialization.py", line 230, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/torch/serialization.py", line 211, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'cv-ab-seq/atrous_self_weights.pth.tar'
```

Fix: Obtain the model weights or retrain the models from scratch?

</details>


<details>
<summary>13. Failed to perform inference using the antigen model(s)</summary>

Steps to replicate:
```bash
python library_commands.py pdb 4bz1 --model AFP --abh H --abl L --ag A
```

Error:

When trying to get the attentional coefficients for each amino acid from the model an index error occurs:
```
after model
in ag visual
writing to visualisation file
Called build_the_pdb_data
4bz1
H
L
name A
pos1 tensor(2)
pos2 tensor(7)
Traceback (most recent call last):
  File "library_commands.py", line 191, in <module>
    main()
  File "library_commands.py", line 184, in main
    arguments["--abh"], arguments["--abl"])
  File "library_commands.py", line 170, in process_single_pdb
    print_ag_weights(out_file_name=pdb_name, model=model)
  File "/home/phil/Documents/GitHub/attentive-parapred/paratope/visualisation.py", line 452, in print_ag_weights
    weights = weights[pos2][0]
IndexError: invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number
```

Fix: I am unsure of what the embeddings of the model represent and how to retrieve these weights so am unsure how to fix this error.

</details>

<details>
<summary>14. :heavy_check_mark: How to get the probability of being in the paratope for each residue</summary>

Perform inference for a given PDB file/complex using the Fast-Parapred model:
```bash
python library_commands.py pdb 4bz1 --model FP --abh H --abl L
```

Get just the probability of being within the paratope for each residue:
```python
import pandas as pd

pdb_fname = '4bz1'

with open(pdb_fname) as f:
    lines = [line.rstrip() for line in f]

probs_dict = {'residue' : [], 'paratope_probability' : []}

for line in lines:
    if line.startswith('ATOM') or line.startswith('HETATM'):
      probs_dict['residue'].append(line[16:20])
      probs_dict['paratope_probability'].append(line[60:66])

probs_df = pd.DataFrame(probs_dict)
# Filter for residues > 95% probability of being within the paratope
probs_df[probs_df['paratope_probability'].astype(float) > 0.95]
```

</details>


<!-- When using a dataset with too few samples

Used precomputed dataset containing too few samples
```
Precomputed dataset found, loading...
Crossvalidation run 1
cdrs torch.Size([6, 32, 34])
ag torch.Size([6, 1269, 28])
Traceback (most recent call last):
  File "/home/pp502/attentive-parapred/paratope/main.py", line 149, in <module>
    run_cv()
  File "/home/pp502/attentive-parapred/paratope/main.py", line 104, in run_cv
    kfold_cv_eval(dataset,
  File "/home/pp502/attentive-parapred/paratope/evaluation.py", line 54, in kfold_cv_eval
    for i, (train_idx, test_idx) in enumerate(kf.split(cdrs)):
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/sklearn/model_selection/_split.py", line 331, in split
    raise ValueError(
ValueError: Cannot have number of splits n_splits=10 greater than the number of samples: n_samples=6.
``` -->

# Epitope prediction

## Training (running `main.py`)

<details>
<summary>15. :heavy_check_mark: No such file or directory: 'data/sabdab_27_jun_95_90.csv'</summary>

Steps to replicate:
```bash
cd epitope
python main.py
```

Error:
```
Traceback (most recent call last):
  File "main.py", line 6, in <module>
    from preprocessing import open_dataset
  File "/home/pp502/attentive-parapred-test/epitope/preprocessing.py", line 19, in <module>
    data_frame = pd.read_csv(DATA_DIRECTORY + CSV_NAME)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/pandas/io/parsers.py", line 686, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/pandas/io/parsers.py", line 452, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/pandas/io/parsers.py", line 946, in __init__
    self._make_engine(self.engine)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/pandas/io/parsers.py", line 1178, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/pandas/io/parsers.py", line 2008, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 674, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] No such file or directory: 'data/sabdab_27_jun_95_90.csv'
```

Fix: Done in [0b113c2](https://github.com/PhilPalmer/attentive-parapred/commit/0b113c293bc0f14072bf0c80793b1f9cb0ab486e) by updating the data dir and PDB paths in `constants.py`

</details>


<details>
<summary> 16. Changes made to the paratope prediction for the newer version of PyTorch have not been made for the epitope prediction</summary>

Steps to replicate:
```bash
cd epitope
python main.py
```

Error:
```
Traceback (most recent call last):
  File "main.py", line 18, in <module>
    from evaluation import *
  File "/home/pp502/attentive-parapred-test/epitope/evaluation.py", line 5, in <module>
    np.set_printoptions(threshold=np.nan)
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/numpy/core/arrayprint.py", line 243, in set_printoptions
    opt = _make_options_dict(precision, threshold, edgeitems, linewidth,
  File "/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/numpy/core/arrayprint.py", line 86, in _make_options_dict
    raise ValueError("threshold must be non-NAN, try "
ValueError: threshold must be non-NAN, try sys.maxsize for untruncated representation
```

Fix: Add [these changes](https://github.com/PhilPalmer/attentive-parapred/commit/d6a1740faf834d3439fa0e67a014f7e2ad6040a1) to scripts in the epitope directory

</details>