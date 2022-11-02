# LPKT and LPKT-S tensorflow_version

Source code and data set for the paper Monitoring Student Progress for Learning Process-consistent Knowledge Tracing.

The code is the implementation of LPKT and LPKT-S model, and the data set is the public data set [ASSIST2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect).

If this code helps with your studies, please kindly cite the following publication:
```
to be updated
```

## Dependencies:

- python >= 3.7
- tesorflow-gpu >= 2.0 
- numpy
- tqdm
- utils
- pandas
- sklearn


## Usage

First, download the data file: [2012-2013-data-with-predictions-4-final.csv](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect), then put it in the folder 'data/' 

Then, run data_pre.py to preprocess the data set, and run data_save.py {sequence length} to divide the original data set into train set, validation set and test set. 

`python data_pre.py`


`python data_save.py 50`

Train the model:

`python train_lpkt_s.py {fold}`

For example:

`python train_lpkt_s.py 1`  or `python train_lpkt_s.py 2`

Test the trained the model on the test set:

`python test.py {model_name}`



