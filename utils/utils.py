import os

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

