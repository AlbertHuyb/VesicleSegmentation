## 2D Vesicle Segmentation

!(./static/project.png)[]

Main framework of training and inference code comes from [C-3-Framework](https://github.com/gjy3035/C-3-Framework). The original liscense is included under `./liscense`

### Environment
* Follow [C-3-Framework](https://github.com/gjy3035/C-3-Framework)
	* Python 3.x
	* Pytorch 1.0 (some networks only support 0.4): http://pytorch.org .
	* other libs in `requirements.txt`, run `pip install -r requirements.txt`.
	* Perhaps some packages like h5py need to be installed after above command. `pip install h5py`

### Data Preparation
* `cd ./data`
* Modify settings in `./data/config.py`
* Run `python 0_augment.py` to do augmentation manually.
* Run `python 1_data_prepared.py` to split sugmented data into train, valid, test datasets.

### Training
* Modify settings in `./training/config.py`
* `cd ./training`
* `python train.py`
* Results will be stored at `./training/exp/`. Tensorboard can be used to visualize the result by `tensorboard --logdir=exp --port=6006`

### Pretrained Model
* Pretrained model can be obtained from [google drive](https://drive.google.com/open?id=1Z864m14b8nygVE3Rv-dI1zzXuc5uh7ZJ)
* Download to default position at `./model`

### Inference
`cd ./inference`
1. Synapse locations to input images and masks.
	* By `python 0_synapse_gen.py`
	* Output to `OUTPUT_DIR` defined in `./code/config.py`, including:
		- dir `full`: full input image
		- dir `part`: masked input image (not used)
		- dir `mask`: binary mask
2. Use trained model to predict the heat map.
	* By `python 1_test_all.py`.
	*  Predict with `MODEL_DIR/MODEL_NAME` directed by `./code/config.py`. Output to `OUTPUT_DIR/result`, including:
		- dir `pred`: heat map in npy format
		- dir `mask`: predicted mask by 0.3 threshold
		- dir `gt`: empty
3. Multiply the predicted mask and heat map with input mask.
	* By `python 2_multiply.py`
	* Output to `OUTPUT_DIR/mask_result`, including:
		- dir `pred`: heat map in npy format
		- dir `mask`: predicted binary mask by 0.3 threshold.
4. Use watershed to cut heat map into vesicles. (Two-step Watershed)
	* By `python 3_post_process.py`
	* Output to  `OUTPUT_DIR/wshed_result`, including:
		- dir `data`: segmentations in npy format
		- dir `img`: matplotlib images of segmentation result 
5. Analyze the final 3D bounding box of synapse seg and prepare to display in VAST.
	* By `python 4_final_step.py`
	* Output to `OUTPUT_DIR/vast_volume`

