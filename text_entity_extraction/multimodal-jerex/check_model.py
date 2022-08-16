import json
from operator import sub
import pandas as pd
import ast
import os
import requests

import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TestConfig
from jerex import model, util

model_cpt = torch.load("text_entity_extraction/multimodal-jerex/data/models/docred_joint/joint_multi_instance/model.ckpt")
print(model_cpt['hyper_parameters'])