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

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)


@hydra.main(config_name='test', config_path='configs/docred_joint')
def inference(cfg: TestConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.dataset, 'test_path')
    util.config_to_abs_paths(cfg.dataset, 'save_path')
    util.config_to_abs_paths(cfg.dataset, 'csv_path')
    util.config_to_abs_paths(cfg.dataset, 'types_path')
    util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    # results_df = model.test_on_df(cfg)
    # print(results_df.head())

    # results_df = pd.read_csv(cfg.dataset.csv_path)

    # relations_df = pd.DataFrame(columns=['doc_id','head','had_type','tail','tail_type','relation'])

    # entity_linking_df = generate_entity_linking_df(cfg, results_df)

    # entity_linking_df.to_csv(os.path.join(cfg.dataset.save_path,"articles_entity_linking.csv"),index=False)

    entity_linking_df = pd.read_csv(cfg.dataset.csv_path)

    entity_linking_df.fillna("",inplace=True)

    df_json = entity_linking_df.to_json(orient="records")
    df_json = json.loads(df_json)
    
    response = requests.post("http://blink:5000/df_link", json=df_json)

    print(type(response.json()))

    df = pd.json_normalize(response.json(), max_level=0)

    print(df)
    print(df.info())

    df.to_csv(os.path.join(cfg.dataset.save_path,"articles_entity_linked.csv"),index=False)

    # for idx, row in results_df.iterrows():
    #     doc_id = row['doc_id']
    #     relations = ast.literal_eval(row['relations'])
    #     tokens = ast.literal_eval(row['tokens'])
    #     entities = []
    #     for relation in relations:
            # head_entity = " ".join(tokens[relation['head_span'][0]:relation['head_span'][1]])
            # if head_entity not in entities:
            #     print("Entity:")
            #     print(" ".join(tokens[relation['head_span'][0]:relation['head_span'][1]]))
            #     print(" ".join(tokens[relation['head_span'][0]-100:relation['head_span'][0]]))
            #     print("\n")
            #     print(" ".join(tokens[relation['head_span'][1]:relation['head_span'][1]+100]))
            #     print("\n")
            #     entities.append(head_entity)
            # relations_df.loc[-1] = [doc_id,relation['head'],relation['head_type'],relation['tail'],relation['tail_type'],relation['relation']]  # adding a row
            # relations_df.index = relations_df.index + 1  # shifting index
            # relations_df = relations_df.sort_index()  # sorting by index

    # print(relations_df.head())

    # results_df = model.test_on_fly(cfg)


    # if results_df is not None:
    #     nodes_df, relations_df, triples_df = generate_neo4j_dfs(cfg,results_df)
    #     idx_to_node = dict(zip(nodes_df.node_id, nodes_df.node_name))
    #     idx_to_relation = dict(zip(relations_df.relation_id, relations_df.relation))

    #     for idx, (subject, relation, object) in triples_df.iterrows():
    #         print(idx_to_node[subject],idx_to_relation[relation],idx_to_node[object])


def generate_entity_linking_df(cfg: TestConfig,results_df):

    entities_linking_df = pd.DataFrame(columns=['doc_id','mention', 'mention_type','context_left','context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        relations = ast.literal_eval(row['relations'])
        tokens = ast.literal_eval(row['tokens'])
        entities = []
        for relation in relations:
            head_entity = " ".join(tokens[relation['head_span'][0]:relation['head_span'][1]])
            if head_entity not in entities:
                print("Head Entity:")
                print(head_entity)
                left_context = " ".join(tokens[relation['head_span'][0]-100:relation['head_span'][0]])
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['head_span'][1]:relation['head_span'][1]+100])
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, head_entity, relation['head_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(head_entity)

            tail_entity = " ".join(tokens[relation['tail_span'][0]:relation['tail_span'][1]])
            if tail_entity not in entities:
                print("Tail Entity:")
                print(tail_entity)
                left_context = " ".join(tokens[relation['tail_span'][0]-100:relation['tail_span'][0]])
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['tail_span'][1]:relation['tail_span'][1]+100])
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, tail_entity, relation['tail_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(tail_entity)

    print(entities_linking_df.head())
    return entities_linking_df


def generate_neo4j_dfs(cfg: TestConfig, results_df):
    json_path = cfg.dataset.types_path

    with open(json_path, 'r') as f:
        types_dict = json.load(f)

    relations_types = types_dict['relations']

    relations_df = pd.DataFrame(columns=['relation','relation_id'])
    relation_to_idx = {}

    for relation,relation_dict in relations_types.items():
        relations_df.loc[-1] = [relation_dict['verbose'],relation.strip()] # adding a row
        relations_df.index = relations_df.index + 1  # shifting index
        relations_df = relations_df.sort_index()  # sorting by index
        relation_to_idx[relation_dict['verbose']] = relation.strip()

    # nodes_df = pd.DataFrame(columns=['name','entity_type'])
    triples_df = pd.DataFrame(columns=['subject','relation','object'])

    head_nodes_df = results_df[['doc_id','head', 'head_type']]
    head_nodes_df = head_nodes_df.drop_duplicates()
    head_nodes_df.columns = ['doc_id','node_name', 'entity_type']
    tail_nodes_df = results_df[['doc_id','tail', 'tail_type']]
    tail_nodes_df = tail_nodes_df.drop_duplicates()
    tail_nodes_df.columns = ['doc_id','node_name', 'entity_type']
    nodes_df = pd.concat([head_nodes_df, tail_nodes_df]).reset_index(drop=True)

    nodes_df['node_id'] = nodes_df.index
    node_to_idx = dict(zip(nodes_df.node_name, nodes_df.node_id))

    for idx, (subject, subject_type, object, object_type,relation) in results_df.iterrows():
        triples_df.loc[-1] = [node_to_idx[subject],relation_to_idx[str(relation)],node_to_idx[object]] # adding a row
        triples_df.index = triples_df.index + 1  # shifting index
        triples_df = triples_df.sort_index()

    print(nodes_df.head())
    print(triples_df.head())
    print(relations_df.head())

    return nodes_df, relations_df, triples_df
    
if __name__ == '__main__':

    inference()
    