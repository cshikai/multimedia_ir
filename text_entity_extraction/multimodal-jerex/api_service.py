import json
from operator import sub
from telnetlib import Telnet
import pandas as pd
import ast
import os
import requests

from fastapi import FastAPI, Request

import torch

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TestConfig
from jerex import model, util

app = FastAPI()

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)


# def load_configs():

initialize(config_path="configs/docred_joint", job_name="app")
cfg = compose(config_name="test")
print(OmegaConf.to_yaml(cfg))

util.config_to_abs_paths(cfg.dataset, 'test_path')
util.config_to_abs_paths(cfg.dataset, 'save_path')
util.config_to_abs_paths(cfg.dataset, 'csv_path')
util.config_to_abs_paths(cfg.dataset, 'types_path')
util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')
util.config_to_abs_paths(cfg.misc, 'cache_path')

configs = cfg

print(configs)

def single_inference(cfg: TestConfig, docs):
    results_df = model.api_call_single(cfg, docs)
    df_json = results_df.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/single_inference")
async def link(request: Request):
    dict_str = await request.json()
    json_dict = dict_str

    # data = json_dict['text'].split("\n")
    # data = [text for text in data if len(text) > 5]
    data = [json_dict['text']]
    print(data)
    test_configs = configs
    json_string = single_inference(test_configs, data)

    json_string = json.loads(json_string)

    return json_string

@app.post("/df_link")
async def link(request: Request):
    df_dict_str = await request.json()
    # df_json = json.dumps(df_dict_str)
    # df = pd.read_json(df_json, orient="records")
    df = pd.json_normalize(df_dict_str, max_level=0)
    print(df.head())
    print(df.info())
    
    df.to_csv(os.path.join(configs.dataset.save_path,"temp.csv"),index=False)

    results_df = model.test_on_df(configs)
    print(results_df.head())

    try:
        os.remove(os.path.join(configs.dataset.save_path,"temp.csv"))
    except:
        print("cannot remove temp")

    df_json = results_df.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json

