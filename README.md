## 2D Vesicle Segmentation
### Problem: 
* a 2D segmentation problem.
### Challenge: 
* need more annotated data to do training work.
### Pipeline:
1. Synapse locations to input images and masks.
	* By ~/EM/VesicleSeg/synapse_gen.py
	* Output to /mnt/pfister_lab2/yubin/vesicleNew/vesicle_18_track. 
		- Full: full input image
		- Part: masked input image (not used)
		- Mask: binary mask
2. Use trained model to predict the heat map.
	* By  code on rc. Need to be archived.
	* Output to /mnt/pfister_lab2/yubin/vesicleNew/vesicle_18_track/result.
		- Pred: heat map in npy format
		- Mask: predicted mask by 0.3 threshold
		- Gt: empty
3. Multiply the predicted mask and heat map with input mask.
	* By ~/EM/VesicleSeg/multiply.py
	* Output to /mnt/pfister_lab2/yubin/vesicleNew/vesicle_18_track/mask_result
		Pred: heat map in npy format
Mask: predicted mask by 0.3 threshold.
Use watershed to cut heat map into vesicles. (Two-step Watershed)
By ~/EM/VesicleSeg/NeuroG/post_process.py
Output to  /mnt/pfister_lab2/yubin/vesicleNew/vesicle_18_track/wshed_result
Data: segmentations in npy format
Img: matplotlib images of segmentation result 
Analyze the final 3D bounding box of synapse seg and prepare to display format.
By ~/EM/VesicleSeg/final_step.py

