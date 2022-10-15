import re
import ast
import pandas as pd

def generate_nodes_df(entities_df: pd.DataFrame) -> pd.DataFrame:

    nodes_df = pd.DataFrame(columns=['doc_id', 'entity', 'mention', 'mention_span','metadata'])

    for idx, row in entities_df.iterrows():
        metadata = {}
        metadata['mention'] = row['mention']
        metadata['mention_type'] = row['mention_type']
        metadata['entity'] = row['entity_names']
        metadata['entity_ID'] = row['entity_link']

        nodes_df.loc[-1] = [row['doc_id'], row['entity_names'],
                               row['mention'], row['mention_span'],metadata]  # adding a row
        nodes_df.index = nodes_df.index + 1  # shifting index
        nodes_df = nodes_df.sort_index()  # sorting by index

    return nodes_df

def generate_relations_df(relations_df: pd.DataFrame) -> pd.DataFrame:

    new_relations_df = pd.DataFrame(columns=['doc_id','subject','relation','object'])

    for idx,row in relations_df.iterrows():
        if type(row['relations']) == str:
            relations = ast.literal_eval(row['relations'])
        else:
            relations = row['relations']

        for relation in relations:
            new_relations_df.loc[-1] = [row['doc_id'], relation['head'],
                                relation['relation'], relation['tail']]  # adding a row
            new_relations_df.index = new_relations_df.index + 1  # shifting index
            new_relations_df = new_relations_df.sort_index()  # sorting by index

    return new_relations_df

def resolve_relations_entities(relations_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    
    resolved_relations_df = pd.DataFrame(columns=['doc_id','subject','relation','object'])

    for idx, row in relations_df.iterrows():
        sub_entity_row = nodes_df.loc[(nodes_df['mention'] == row['subject']) & (nodes_df['doc_id'] == row['doc_id'])]
        sub_entity = sub_entity_row['entity'].values[0] if sub_entity_row['entity'].values.size > 0 else ''
        obj_entity_row = nodes_df.loc[(nodes_df['mention'] == row['object']) & (nodes_df['doc_id'] == row['doc_id'])]
        obj_entity = obj_entity_row['entity'].values[0] if obj_entity_row['entity'].values.size > 0 else ''

        if sub_entity and obj_entity:
            resolved_relations_df.loc[-1] = [row['doc_id'], sub_entity,
                                    row['relation'], obj_entity]  # adding a row
            resolved_relations_df.index = resolved_relations_df.index + 1  # shifting index
            resolved_relations_df = resolved_relations_df.sort_index()  # sorting by index
        
    return resolved_relations_df

if __name__ == '__main__':

    linked_entities_df = pd.read_csv("data/text_articles_entity_linked.csv")
    relations_df = pd.read_csv("data/text_test_jerex.csv")

    nodes_df =  generate_nodes_df(linked_entities_df)
    relations_df = generate_relations_df(relations_df)
    relations_df = resolve_relations_entities(relations_df, nodes_df)

    nodes_df.to_csv("data/nodes.csv", index=False)
    relations_df.to_csv("data/relations.csv", index=False)
