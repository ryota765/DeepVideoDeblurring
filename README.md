# Implementation of Deep Video Deblurring in Keras

This is a re-implementation code of [Deep Video Deblurring for Hand-held Cameras](https://openaccess.thecvf.com/content_cvpr_2017/papers/Su_Deep_Video_Deblurring_CVPR_2017_paper.pdf) in Keras.  
Given a stack of pre-aligned input frames, this network predicts a sharper central image.  
* [Author implementation in matlab](https://github.com/shuochsu/DeepVideoDeblurring)

## Getting Started

### Download scripts

Clone this repository and add some additional directories.  
Your directory should be as shown below. 

```
(parent directory)/
- preprocess.py
- train.py
- evaluation.py
- utils/
-- metrics.py
-- model.py
- data/
-- gt_data
-- input_data
-- id_list
-- output
```

### Download datasets

The dataset can be downloaded from [Github of author implementation](https://github.com/shuochsu/DeepVideoDeblurring).  
This dataset includes frames from 71 videos. (61 for training, 10 for testing)  
Place the downloaded dataset under data directory as shown below.  

```
data/
- DeepVideoDeblurring_Dataset/
-- qualitative_datasets
-- quantitative_datasets/
- gt_data
- input_data
- id_list
- output
```

## Running the code

### Preprocessing

Data augmentation and resizing are done by the code below.  
List including paths of train and test images are also generated under data/id_list directory by this script.  

```
python preprocess.py
```

### Training

Training the model is done by this script.  
Check out several flags which is available in train.py
Figure of training history will be generated under data/output directory.  

```
python train.py --n_epochs=4 
```

### Prediction

Generate clear image using weights of a trained model.  

```
python predict.py --data_dir=data --model_path=data/output/model_weights.h5
```

### Evaluation

Evaluation of the trained model.  
Figure of evaluation results will be generated under data/output directory.  

```
python evaluation.py
```

## Author

* **Ryota Nomura** - *Initial work* - [HomePage](http://ryota-n.info/)


## Acknowledgments

* [Deep Video Deblurring for Hand-held Cameras](https://openaccess.thecvf.com/content_cvpr_2017/papers/Su_Deep_Video_Deblurring_CVPR_2017_paper.pdf) - Original Paper
* [Code by author](https://github.com/shuochsu/DeepVideoDeblurring) - Implementation in matlab made by the authors

