# MRI-INR - MRI Image Reconstruction with Modulated SIREN Networks

This is the repository for the Applied Deep Learning in Medicine practical at the chair of AI in Medicine at TUM. 

Group Members:
- Andreas Ehrensberger 
- Jan Rogalka 
- Matteo Wohlrapp


To train the model, simply execute the main.py file. You can see visualizations with `tensorboard --logdir=runs`. 

## Project Overview
This project implements implicit neural representations for MRI images using modulated SIREN (Sinusoidal Representation Networks). It is designed to reconstruct high-quality images from undersampled k-space images.

## Installation
Ensure that you have Python installed on your system (Python 3.8+ recommended). Install all required dependencies by running:

```bash
pip install -r requirements.txt
```

This will install all necessary Python packages as specified in the `requirements.txt` file.

## Configuration
The project uses YAML files for configuration to specify parameters for training and testing. Modify these files to adjust various parameters like dataset paths, network architecture, learning rates, etc. Alternatively, you can adjust all of the parameters through command line arguments. 

Example configuration files are located in the `./configuration` directory:
- `train.yml` for training setup.
- `test.yml` for testing setup.

## Running the Application
To run the application, use the `main.py` script. You can specify a custom configuration file or use the provided examples. Just make sure to specify the execution mode:

**Training:**
```bash
python main.py -c ./configuration/train.yml
```
You can see visualizations of the training by running `tensorboard --logdir=runs`.

**Testing:**
```bash
python main.py -c ./configuration/test.yml
```

The `-c` flag is used to specify the path to the configuration file. <br>
To run the `autoencoder` encoder, you will need to download the VGG-16 pretrained model on ImageNet from [here](https://github.com/Horizon2333/imagenet-autoencoder/tree/main) and add it under `output/model_checkpoints` in the root of the repository.

## Command-Line Arguments
Use the `-h` option to view all available command-line arguments and their descriptions:

```bash
python main.py -h
```

This will display help information for each argument, including default values and choices where applicable.

## Output
All output files, including saved models and reconstructed images, are stored in a subdirectory withing the `output` directory specified by the `--name` argument within the configuration file. This allows for easy organization and retrieval of results from different runs.
