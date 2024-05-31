import argparse
import yaml

def create_parser():
    parser = argparse.ArgumentParser(description='Autoencoder Training Script')
    
    # Add argument for the YAML configuration file
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    
    return parser

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_args():
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    return config
