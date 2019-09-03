## 2D Vesicle Segmentation
### Problem: 
* a 2D segmentation problem.
### Challenge: 
* need more annotated data to do training work.
### PredicTion Pipeline:
1. Synapse locations to input images and masks.
	* By `python ./code/0_synapse_gen.py`
	* Output to `OUTPUT_DIR` defined in `./code/config.py`, including:
		- dir `full`: full input image
		- dir `part`: masked input image (not used)
		- dir `mask`: binary mask
2. Use trained model to predict the heat map.
	* By `python ./code/1_test_all.py`.
	*  Predict with `MODEL_DIR/MODEL_NAME` directed by `./code/config.py`. Output to `OUTPUT_DIR/result`, including:
		- dir `pred`: heat map in npy format
		- dir `mask`: predicted mask by 0.3 threshold
		- dir `gt`: empty
3. Multiply the predicted mask and heat map with input mask.
	* By `python ./code/2_multiply.py`
	* Output to `OUTPUT_DIR/mask_result`, including:
		- dir `pred`: heat map in npy format
		- dir `mask`: predicted binary mask by 0.3 threshold.
4. Use watershed to cut heat map into vesicles. (Two-step Watershed)
	* By `python ./code/3_post_process.py`
	* Output to  `OUTPUT_DIR/wshed_result`, including:
		- dir `data`: segmentations in npy format
		- dir `img`: matplotlib images of segmentation result 
5. Analyze the final 3D bounding box of synapse seg and prepare to display in VAST.
	* By `python ./code/4_final_step.py`
	* Output to `OUTPUT_DIR/vast_volume`

