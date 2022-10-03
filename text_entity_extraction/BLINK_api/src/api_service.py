import ast
import json
import requests
import pandas as pd
from fastapi import FastAPI, Request
from inference import Inference, NewKBEntities

app = FastAPI()


def row_linking(row):

    data_to_link = []

    data = {
        "id": 0,
        "doc_id": row['doc_id'],
        "label": "unknown",
        "label_id": -1,
        "context_left": row['context_left'].lower() if row['context_left'] is not None else "",
        "mention": row['mention'].lower()if row['mention'] is not None else "",
        "context_right": row['context_right'].lower()if row['context_right'] is not None else ""
    }
    data_to_link.append(data)

    print(data_to_link)

    inference = Inference(data_to_link)
    results = inference.run_inference()

    if results:
        entity_id = [row['entity_id'] for row in results['entities']]
    else:
        entity_id = -1

    return entity_id


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/single_inference")
async def link(request: Request):
    dict_str = await request.json()
    json_dict = dict_str

    data = {
        "id": 0,
        "doc_id": json_dict['doc_id'] if 'doc_id' in json_dict.keys() else 0,
        "label": "unknown",
        "label_id": -1,
        "context_left": json_dict['context_left'].lower() if json_dict['context_left'] is not None else "",
        "mention": json_dict['mention'] if json_dict['mention'] is not None else "",
        "context_right": json_dict['context_right'].lower()if json_dict['context_right'] is not None else ""
    }

    data_to_link = []
    data_to_link.append(data)

    inference = Inference(data_to_link)
    results = inference.run_inference()

    json_string = results['entities'][0]
    json_string = json.dumps(json_string)

    return json_string


@app.post("/df_link")
async def link(request: Request):
    df_dict_str = await request.json()
    df_json = json.dumps(df_dict_str)
    df = pd.read_json(df_json, orient="records")
    df = df.reset_index(drop=True)
    print(df.head())
    print(df.info())

    data_to_link = []

    for idx, row in df.iterrows():
        data = {
            "id": idx,
            "doc_id": row['doc_id'],
            "label": "unknown",
            "label_id": -1,
            "context_left": row['context_left'].lower() if row['context_left'] is not None else "",
            "mention": row['mention'].lower()if row['mention'] is not None else "",
            "context_right": row['context_right'].lower()if row['context_right'] is not None else ""
        }

        data_to_link.append(data)

    inference = Inference(data_to_link)
    results = inference.run_inference()

    # entity_ids = [row['entity_id'] for row in results['entities']]
    # results.loc[results["entity_link"] == "Unknown", "entity_link"] = str(-1)
    # results["entity_link"] = pd.to_numeric(results["entity_link"])
    entity_links = [row['entity_link'] for row in results['entities']]
    entity_links = [entity_link.split("curid=")[-1]
                    for entity_link in entity_links]
    entity_names = [row['entity_linked'] for row in results['entities']]
    link_type = [row['link_type'] for row in results['entities']]
    score = [row['entity_confidence_score'] for row in results['entities']]
    embedding = [row['embeddings'] for row in results['entities']]

    # print("Entity IDs length: ", len(entity_ids))

    df_linked = df
    # df_linked['entity_id'] = entity_ids
    df_linked['entity_link'] = entity_links
    df_linked['entity_names'] = entity_names
    df_linked['link_type'] = link_type
    df_linked['score'] = score
    df_linked['embeddings'] = embedding
    df.drop(columns=['context_left', 'context_right', 'mention_span'])

    print(df_linked.head())
    print(df_linked.info())

    df_json = df_linked.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json


@app.post("/link_row")
async def link(request: Request):
    df_dict_str = await request.json()
    df_json = json.dumps(df_dict_str)
    df = pd.read_json(df_json, orient="records")
    print(df.head())
    print(df.info())

    entity_ids = []

    for idx, row in df.iterrows():
        entity_id = row_linking(row)
        entity_ids.append(entity_id)

    df['entity_id'] = entity_ids

    print(df.head())

    df_json = df.to_json(orient="records")
    df_json = json.loads(df_json)

    return df_json


@app.post("/add_entities")
async def link(request: Request):
    df_dict_str = await request.json()
    df_json = json.dumps(df_dict_str)
    df = pd.json_normalize(df_dict_str, max_level=0)
    print(df.head())
    print(df.info())

    entities = []

    for idx, row in df.iterrows():
        data = {
            "idx": row['idx'],
            "text": row['text'],
            "title": row['title'],
            "entity": row['entity']
        }

        entities.append(data)

    entity_adder = NewKBEntities(entities)
    entity_adder.add_entities_to_kb()

    return df_json


# [{"text": " Shearman Chua was born in Singapore, in the year 1996. He is an alumnus of NTU and is currently working at DSTA. ", "idx": "https://en.wikipedia.org/wiki?curid=88767376", "title": "Shearman Chua", "entity": "Shearman Chua"},
# {"text": " The COVID-19 recession is a global economic recession caused by the COVID-19 pandemic. The recession began in most countries in February 2020. After a year of global economic slowdown that saw stagnation of economic growth and consumer activity, the COVID-19 lockdowns and other precautions taken in early 2020 drove the global economy into crisis. Within seven months, every advanced economy had fallen to recession. The first major sign of recession was the 2020 stock market crash, which saw major indices drop 20 to 30% in late February and March. Recovery began in early April 2020, as of April 2022, the GDP for most major economies has either returned to or exceeded pre-pandemic levels and many market indices recovered or even set new records by late 2020. ", "idx": "https://en.wikipedia.org/wiki?curid=63462234", "title": "COVID-19 recession", "entity": "COVID-19 recession"},
# {"text": " The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The novel virus was first identified from an outbreak in Wuhan, China, in December 2019. Attempts to contain it there failed, allowing the virus to spread worldwide. The World Health Organization (WHO) declared a Public Health Emergency of International Concern on 30 January 2020 and a pandemic on 11 March 2020. As of 15 April 2022, the pandemic had caused more than 502 million cases and 6.19 million deaths, making it one of the deadliest in history. ", "idx": "https://en.wikipedia.org/wiki?curid=62750956", "title": "COVID-19 pandemic", "entity": "COVID-19 pandemic"}]
