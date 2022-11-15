# LPKT-S and LPKT tensorflow_version

Source code and data set for the paper Monitoring Student Progress for Learning Process-consistent Knowledge Tracing.

The code is the implementation of LPKT and LPKT-S model, and the data set is the public data set [ASSIST2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect).

If this code helps with your studies, please kindly cite the [following publication](https://ieeexplore.ieee.org/document/9950313):
```
@ARTICLE{9950313,  author={Shen, Shuanghong and Chen, Enhong and Liu, Qi and Huang, Zhenya and Huang, Wei and Yin, Yu and Su, Yu and Wang, Shijin},  journal={IEEE Transactions on Knowledge and Data Engineering},   title={Monitoring Student Progress for Learning Process-Consistent Knowledge Tracing},   year={2022},  volume={},  number={},  pages={1-15},  doi={10.1109/TKDE.2022.3221985}}
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



