'''
Directory containing the raw data, containing:
    - im: images of synapse vesicles
    - gt: gt mask of synapse vesicles with the same name of corresponding images
'''
RAW_DATA = '/home/huyb/EM/CrowdCounting/ProcessedData/origin/vesicle_4nm'


'''
Directory containing the augmented data, containing:      
    - im: images of synapse vesicles        
    - gt: gt mask of synapse vesicles with the same name of corresponding images        
Augmentations can be checked at 0_augment.py
'''
AUG_DIR = '/home/huyb/EM/CrowdCounting/ProcessedData/origin/augmentation'

'''
Directory containing the prepared data for training, containing:
    - train_data: data for training stage
    - valid_data: data for valid stage 
    - test_data: data for test stage
Split ratio can be checked at 1_data_prepared.py
'''
DATA_DIR = '/home/huyb/EM/CrowdCounting/ProcessedData/Vesicles2'
