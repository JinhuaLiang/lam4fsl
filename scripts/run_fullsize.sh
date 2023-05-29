WORK_DIR="PATH/TO/DIR"
STORAGE_DIR="PATH/TO/STORAGE"  # `dataset dir` and `pretrained weight pth` should be under this dir
# Set variables (The below are receommended settings)
# |            	| esc50      	| fsdkaggle18k 	| fsd_fs     	|
# |------------	|------------	|--------------	|------------	|
# | EXPERIMENT 	| sl_fewshot 	| sl_fewshot   	| ml_fewshot 	|
# | DATABASE   	| esc50      	| fsdkaggle18k 	| fsd_fs     	|
# | N_TASK     	| 100        	| 100          	| 50         	|
# | N_CLASS    	| 15         	| 10           	| 15         	|
# | N_QUERIES  	| 30         	| 50           	| 5          	|
# | N_SUPPORTS 	| 10         	| 20           	| 5          	|
# | N_EPOCHS   	| 20         	| 20           	| 20         	|

EXPERIMENT=esc50_fullsize_evaluation  # [esc50_fullsize_evaluation, fsdkaggle18k_fullsize_evaluation]
MODEL_NAME=ms_clap
N_SUPPORTS=1,3,5,10,15,20,25,32  #1,3,5,10,15,20,25,32
FINE_TUNE=True
N_EPOCHS=40,45,50 # 40,50,60  #10,15,20,25,30,35,40,45,50,60,70,80,100
ADAPTER=xattention # match, xattention
LEARNING_RATE=0.0001,0.0005,0.001 #0.0001,0.001
TRAIN_A=True

CUDA_VISIBLE_DEVICES=-1 python3 ${WORK_DIR}/${EXPERIMENT}.py \
storage_pth=${STORAGE_DIR} \
model_name=${MODEL_NAME} \
fewshot.fine_tune=${FINE_TUNE} \
fewshot.adapter=${ADAPTER} \
fewshot.n_supports=${N_SUPPORTS} \
fewshot.train_epochs=${N_EPOCHS} \
fewshot.learning_rate=${LEARNING_RATE} \
fewshot.train_a=${TRAIN_A}
