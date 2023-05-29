WORK_DIR="PATH/TO/DIR"
STORAGE_DIR="PATH/TO/STORAGE"  # `dataset dir` and `pretrained weight pth` should be under this dir

##### Finetune model #####
MODEL_NAME=ms_clap
N_SUPPORTS=1,3,5,10,15,16,20,32  # number of examples for training 
N_EPOCHS=200
LEARNING_RATE=0.001
BATCH_SIZE=40

python3 ${WORK_DIR}/finetune.py \
storage_pth=${STORAGE_DIR} \
model_name=${MODEL_NAME} \
fewshot.n_supports=${N_SUPPORTS} \
fewshot.train_epochs=${N_EPOCHS} \
fewshot.learning_rate=${LEARNING_RATE} \
fewshot.batch_size=${BATCH_SIZE}
