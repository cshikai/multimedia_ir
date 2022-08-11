import requests
import json
from typing import List, Dict
import pandas as pd
import ast

def predict_jerex(dataset: Dict):
    response = requests.post('http://0.0.0.0:8080/df_link', json = dataset)
    
    #ipdb.set_trace()
    # print([doc for doc in response.iter_lines()])
    response = response.json()
    # print(type(response))
    # df_json = json.loads(response)
    # print(type(df_json))
    df = pd.json_normalize(response, max_level=0)

    print(df.head())
    print(df.info())

    df.to_csv("data/test_jerex.csv",index=False)

    return df

def predict_blink(dataset: Dict):

    response = requests.post('http://0.0.0.0:5050/df_link', json = dataset)
    # df_json = json.dumps(response.json())
    # df = pd.read_json(df_json, orient="records")

    # print(df.head())
    # print(df.info())

    df = pd.json_normalize(response.json(), max_level=0)

    print(df)
    print(df.info())

    # df.to_csv("data/articles_entity_linked_exact.csv",index=False)

    return df

def generate_entity_linking_df(results_df):

    entities_linking_df = pd.DataFrame(columns=['doc_id','mention', 'mention_type','context_left','context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['relations']) == str:
            relations = ast.literal_eval(row['relations'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            relations =  row['relations']
            tokens = row['tokens']
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


if __name__ == '__main__':
    # dataset = {"sentence":"Apple computers"}
    # dataset = {
    #             "context_left": "Who manufactured the".lower(),
    #             "mention": "Oerlikon cannon",
    #             "context_right": "".lower(),
    #           }
    # dataset = {
    #         "text": '''Kimi Raikkonen said on Thursday he was looking forward to retirement after 19 seasons and 349 races in Formula One and would be less emotional about it than his wife Minttu.
    #         Ferrari's 2007 world champion, now driving for Alfa Romeo, has one more race in Abu Dhabi on Sunday before he calls it quits at the age of 42.
    #         "I'm looking forward to get the season done," the Finnish 'Iceman', who made his Formula One debut in 2001, told reporters at Yas Marina.
    #         "It's nice that it comes to an end and I'm looking forward to the normal life after.
    #         "I think for sure my wife will be more emotional about it," added the poker-faced winner of 21 races.
    #         "I doubt that the kids will really care either way, I think they will find other things to do that are more interesting. They like coming to a warm country and be in a pool and other things but it’s nice to have them here."
    #         Raikkonen announced in September he would be retiring at the end of the season, with the Swiss-based team signing compatriot Valtteri Bottas as his replacement from Mercedes.
    #         Alfa Romeo will be marking his final race with a special livery tribute on the side of his car declaring "Dear Kimi, we will leave you alone now".
    #         Raikkonen famously uttered the phrase "Just leave me alone, I know what I'm doing" over the radio while heading to victory in Abu Dhabi with Lotus in 2012, a comment that spawned a range of merchandise and social media memes.
    #         Ever popular with the fans, the Finn said he was looking forward to life without a rigid schedule.
    #         "Right now I’m not looking at anything apart from finishing the year," he said.
    #         "We’ll see if there’s some interesting things that comes out, if it makes sense maybe I’ll do it, but I have zero plans right now."'''
    #     }

    articles_df = pd.read_csv("data/articles_2.csv")
    dataset = {"text": articles_df['text'].iloc[0]}

    entity_linking_df = pd.read_csv("data/articles_entity_linking.csv")

    df_json = articles_df.to_json(orient="records")
    df_json = json.loads(df_json)
    jerex_results = predict_jerex(df_json)
    # jerex_results = predict_jerex(dataset)

    entity_linking_df = generate_entity_linking_df(jerex_results)

    print(entity_linking_df)

    df_json = entity_linking_df.to_json(orient="records")
    df_json = json.loads(df_json)

    blink_results = predict_blink(df_json)
