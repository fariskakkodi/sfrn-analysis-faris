'''
https://huggingface.co/models
Most used:

bert-base-uncased
distilbert-base-uncased
roberta-base
google/electra-small-discriminator
YituTech/conv-bert-base
'''

#TRAIN_FILE_PATH = ['./data/debug.csv']
#TEST_FILE_PATH = ['./data/debug.csv']

TRAIN_FILE_PATH = ['./data/train_new_data.csv']
TEST_FILE_PATH = ['./data/test_new_data.csv']
VAL_FILE_PATH = ['./data/validation_new_data.csv']

### CHANGE: added path to istudio_dict_init.py ###
#ISTUDIO_DATA_DICTS = ['./data/smk6961/ASAG/istudio_dict_init.py']
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'

TAG2ID = {'0': 0, '1': 1, '2': 2}

hyperparameters = dict(
    train_id="0204_IS_SFRN_pretrain_model",
    model_name="bert-base-uncased",
    num_labels = 3,
    max_length = 128,
    random_seed=23, # 23， 123
    data_split=0.2,
    lr=1e-6,
    epochs=3,
    weight_decay=0.01,
    max_norm = 1, 
    WARMUP_STEPS=0.05,
    hidden_dropout_prob=0.2,
    GRADIENT_ACCUMULATION_STEPS = 2,
    # model
    hidden_dim=768, # 768
    mlp_hidden=256,
    )
# wandb config

PROJECT_NAME = "UseYourProject"
ENTITY="Use you entity"


config_dictionary = dict(
    #yaml=my_yaml_file,
    params=hyperparameters,
)

### CHANGE: moved dicts to istudio_dict_init.py ###
