from FinnalConfig import config
import os
import json
import torch

def save_model(model, model_name, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model_save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    config_save_path = os.path.join(save_dir, f'{model_name}_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    print(f"Configuration saved to {config_save_path}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    for key, value in config_data.items():
        if not key.endswith('__'):
            setattr(config, key, value)
        if key == 'model__name__':
            print(f"The configuration of {value} is loaded")
    