import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DConvert')

def main():
    # Load configuration
    config, _ = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)

    #scripted_model = torch.jit.script(model)

    # Save the scripted model
    #scripted_model.save("/tmp/custom_model.pt")

    # Switch the model to eval model
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 2, 128, 128)

    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("traced_model.pt")

if __name__ == '__main__':
    main()
