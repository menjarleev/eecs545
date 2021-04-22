# PhotometricGAN
## create image dataset from ```.mat``` dataset
```bash
python ./data/mat2np.py --dataset_root YOUR_DATASET_ROOT --mat_dir MAT_DIR --img_dir IMG_DIR
```
This script creates a grayscale jpg image dataset on ```YOUR_DATASET_ROOT/IMG_DIR``` using mat dataset ```YOUR_DATASET_ROOT/MAT_DIR```.
For example, your dataset is stored in ```/home/ubuntu/data/BottleData```, then ```YOUR_DATASET_ROOT=/home/ubuntu/data```, ```MAT_DIR=BottleData```,
and the new generated dataset is stored in ```/home/ubuntu/data/IMG_DIR```
* note: please remove redundant ```.mat``` files in folder ```YOUR_DATASET_ROOT/MAT_DIR/PHASE/```. For example, you need to manually remove files in
```/home/ubuntu/data/BottleData/Training_Data_128/*.mat``` by your own. In this way, only ```/home/ubuntu/data/BottleData/Training_Data_128/Bottle_95/``` contains 
```input.mat``` and ```output_\d+``` files.

## run code
```bash
main.py --test --test_step 50 --dataset_root YOUR_DATASET_ROOT --num_resblock 12 
--train --max_step 500 --validation_interval 100 --log_interval 100 
--lr 0.00001 --validation --optim_name Adam --save_result --batch_size 8 --batch_size_eval 8
```
Args can be found in ```option/option.py```

## configuration
* TODO add configuration files