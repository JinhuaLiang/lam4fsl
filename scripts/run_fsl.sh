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
EXPERIMENT=sl_fewshot  # [sl_fewshot, ml_fewshot]
MODEL_NAME=ms_clap
DATABASE=esc50
N_TASK=3200
N_CLASS=12
N_QUERIES=5

N_SUPPORTS=5
FINE_TUNE=False
N_EPOCHS=0
ADAPTER=match # match
TRAIN_A=False

python3 ${EXPERIMENT}.py \
storage_pth=${STORAGE_DIR} \
model_name=${MODEL_NAME} \
database=${DATABASE} \
fewshot.n_task=${N_TASK} \
fewshot.n_class=${N_CLASS} \
fewshot.n_queries=${N_QUERIES} \
fewshot.fine_tune=${FINE_TUNE} \
fewshot.adapter=${ADAPTER} \
fewshot.n_supports=${N_SUPPORTS} \
fewshot.train_epochs=${N_EPOCHS} \
fewshot.train_a=${TRAIN_A}
