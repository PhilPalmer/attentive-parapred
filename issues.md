# General

I encountered a number of issues trying to run the models. I have made a note of and characterised the issues here

:heavy_check_mark: = issue now resolved

1) Was this code used for [Attentive cross-modal paratope prediction](https://arxiv.org/pdf/1806.04398) then improved upon for [Neural message passing for joint paratope-epitope prediction](https://arxiv.org/pdf/2106.00757.pdf)?

2) No documentation for how to run the model(s)? (See [#2](https://github.com/andreeadeac22/attentive-parapred/issues/2))

3) What are the inputs? Do both the antibody and antigen need to be input as a complex?

# Paratope prediction

I encountered a number of issues with installing and running the paratope prediction. It's possibly that many of these issues are caused by incorrect installation. Some documentation regarding installation would help solve the problem

## Installation

<details>
<summary>4. :heavy_check_mark: `error: package directory 'aid25' does not exist` </summary>

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
<summary>5. :heavy_check_mark: Coudln't install packages such as sci-kit learn </summary>

Error:

Possibly an issue with my development environment and not the package:
```
error: Setup script exited with error: Command "g++ -pthread -B /home/pp502/.conda/envs/parapred/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/numpy/core/include -I/home/pp502/.conda/envs/parapred/lib/python3.8/site-packages/numpy/core/include -I/home/pp502/.conda/envs/parapred/include/python3.8 -c sklearn/cluster/_dbscan_inner.cpp -o build/temp.linux-x86_64-3.8/sklearn/cluster/_dbscan_inner.o -MMD -MF build/temp.linux-x86_64-3.8/sklearn/cluster/_dbscan_inner.o.d" failed with exit status 1
```

Fix: Manually installed troublesome packages with conda and commented the installs out from my local version of the `setup.py` script

</details>

## Running `fast_parapred`

<details>
<summary>6. :heavy_check_mark: `ModuleNotFoundError: No module named 'constants'` </summary>

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
<summary>7. :heavy_check_mark: `No such file or directory: 'data/sabdab_27_jun_95_90.csv'` </summary>

Steps to replicate:
```bash
python paratope/library_commands.py --help
```

Error: The SAbDab CSV file cannot be found

Fix: Run the `library_commands.py` script is run in the `paratope` directory so that the SAbDab CSV file can be found. It may also be possible to change the location of this file or make it a parameter in the `constants.py` file or make it an input parameter, however, this is slightly more difficult and I couldn't get it to work instantly

</details>


<details>
<summary>8. :heavy_check_mark: `Attempted relative import beyond top-level package` </summary>

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
<summary>9. Model files cannot be found</summary>

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

## Running `main.py`

To rerun the analysis (hopefully regenerate the models )

<details>
<summary>9. Failed to run cross validation for all complexes</summary>

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

Fix: I am unsure of the exact cause of this error and how to fix it

</details>

<details>
<summary>11. Failed to do full run for subset of complexes</summary>

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

Fix: I am unsure of the exact cause of this error and how to fix it

</details>

# Epitope prediction
