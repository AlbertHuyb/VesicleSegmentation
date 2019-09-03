'''
INPUT SEG DIR
Containing:
    images location for the whole volume
Example:
    /mnt/pfister_lab2/donglai/data/JWR/JWR_local-butterfly_dw.json

    {"sections": 
        ["/n/pfister_lab2/Lab/adisuis/JWR15_EM/
          JWR15_W01_to_W12_take4_fullres_tiles_normalized0.1_filtered/
          0306_W02_Sec001_montaged/
          0306_W02_Sec001_montaged_tr{row}-tc{column}.png", 
          ...
'''
SEG_NAME = '../data/JWR_local-butterfly_dw.json'

'''
INPUT MASK DIR
Containing:
    binary masks for the synapses with the same size as SEG_NAME images
Example:
    /mnt/pfister_lab2/vcg_connectomics/JWR15/axon_32nm/cell53/
'''
MASK_DIR = '/mnt/pfister_lab2/vcg_connectomics/JWR15/axon_32nm/cell53/'

'''
SYNAPSE LOCATION FILE
Containing:
    locations of synapses
Example:
    /home/huyb/EM/VesicleSeg/cell53_syn.txt

    24776  26511  693
    13700  24596  1429
    11348  24080  1578
    ...

    You can try these locations in neuroglancer to determine the coordinate order 
'''
LOCATION_FILE = '../data/cell53_syn.txt'

'''
DATA OUTPUT PROJECT DIR
After the  whole procedure, will contain:
    ## synapse input images generated from location annotations
    - full: dir containing 30 layers of 256x256 images around each synapse
    - part: dir containing 30 layers of 256x256 masked images around each synapse
    - mask: dir containing 30 layers of 256x256 binary masks for each synapse
    ## prediction from NN
    - result: dir containing predicted heat map for each 
    - maks-result: dir containing masked predictions
    ## result after post processing
Example: 
    /mnt/pfister_lab2/yubin/vesiclesNew/vesicle_18_track_1gpu
'''
OUTPUT_DIR = '/mnt/pfister_lab2/yubin/vesiclesNew/vesicle_18_track_1gpu'

''' 
NN MODEL DIR
'''
MODEL_DIR = '../model/'

MODEL_NAME = 'all_ep_199_mae_1270.8_mse_2142.9.pth'


