from elasticsearch import Elasticsearch, RequestError, helpers

import pandas as pd
import yaml
from tqdm import tqdm


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

ELASTIC_URL = config['ELASTICSEARCH']['URL']
INDEX_NAME = config['ELASTICSEARCH']['INDEX_NAME']
ELASTIC_USERNAME = config['ELASTICSEARCH']['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTICSEARCH']['ELASTIC_PASSWORD']

client = Elasticsearch(ELASTIC_URL,  # ca_certs="",
                       verify_certs=False,
                       basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

mappings = config['MAPPING']


def es_create_index_if_not_exists(es, index):
    """Create the given ElasticSearch index and ignore error if it already exists"""
    try:
        es.indices.create(index=index, body=mappings)
    except RequestError as ex:
        if ex.error == 'resource_already_exists_exception':
            print("Index already exists!!")
            pass  # Index already exists. Ignore.
        else:  # Other exception - raise it
            raise ex


if __name__ == '__main__':

    col_headers = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country_code',
                   'cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'modification_date']
    df = pd.read_csv(config['geo_file'], sep='\t', names=col_headers)
    col_headers = ['alternateNameId', 'geonameid', 'isolanguage', 'alternate name',
                   'isPreferredName', 'isShortName', 'isColloquial', 'isHistoric', 'from', 'to']
    alt_names = pd.read_csv(config['alt_names'], sep='\t', names=col_headers)
    # Check if indices is create, if yes throw exception
    es_create_index_if_not_exists(client, INDEX_NAME)

    # Translate feature class into numerical value based on importance
    # Used for score weighing subsequently
    feature_map = ['U', 'V', 'H', 'T', 'L', 'R', 'S', 'P', 'A']

    # Filter English preferred names
    en_preferred_alt_names = alt_names[alt_names['isolanguage'] == 'en']
    en_preferred_alt_names = en_preferred_alt_names[en_preferred_alt_names['isPreferredName'] == 1]

    # Augment preferred name into df
    for geo_id in tqdm(en_preferred_alt_names['geonameid'].unique()):
        temp_df = en_preferred_alt_names[en_preferred_alt_names['geonameid'] == geo_id]
        preferred_names = ', '.join(temp_df['alternate name'])
        # Find index of geo entry in main df
        df_index = df[df['geonameid'] == geo_id].index[0]
        # Append preferred name into main df
        df.at[df_index, 'preferrednames'] = preferred_names

    # Perform uploading of Geo data to ES
    actions = []
    count = 0
    for i, rows in tqdm(df.iterrows(), total=len(df)):
        if count == 500:
            count = 0
            helpers.bulk(client, actions)
            actions = []

        loc_dict = rows.dropna().to_dict()
        if 'feature_class' in loc_dict:
            loc_dict['feature_class_num'] = feature_map.index(
                loc_dict['feature_class'])+1  # Ensure value is non-zero
        else:
            loc_dict['feature_class_num'] = 1  # Default to lowest value
        geo_id = loc_dict['geonameid']
        loc_dict.pop('geonameid', None)
        source_dict = {}
        source_dict['_op_type'] = 'index'
        source_dict['_index'] = INDEX_NAME
        source_dict['_id'] = str(geo_id)
        source_dict['_source'] = loc_dict
        actions.append(source_dict)
        count += 1
    # Upload remaining actions
    helpers.bulk(client, actions)
