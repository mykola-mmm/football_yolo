import os
import wandb


def is_running_in_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

def get_roboflow_api():
    if is_running_in_kaggle():
        print("This code is running in a Kaggle notebook")
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("roboflow_api")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("ROBOFLOW_API")

def get_wandb_api():
    if is_running_in_kaggle():
        print("This code is running in a Kaggle notebook")
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret("wandb_api")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("WANDB_API")

def fix_dataset_yaml(dataset):
    YAML_PATH = os.path.join(dataset.location, 'data.yaml')
    with open(YAML_PATH, 'r') as file:
        yaml_data = file.read()

    yaml_data = yaml_data.replace('test: ../test/images', 'test: test/images')
    yaml_data = yaml_data.replace('train: football-players-detection-12/train/images', 'train: train/images')
    yaml_data = yaml_data.replace('val: football-players-detection-12/valid/images', 'val: valid/images')

    with open(YAML_PATH, 'w') as file:
        file.write(yaml_data)

def init_wandb(project="wandb-validate", job_type="validation"):
    WANDB_API = get_wandb_api()
    wandb.require("core")
    wandb.login(key=WANDB_API)
    run = wandb.init(project=project, job_type=job_type)
    return run

def load_model_and_get_path_wandb(run, model_name):
    model_path = run.use_model(name=model_name)
    return model_path