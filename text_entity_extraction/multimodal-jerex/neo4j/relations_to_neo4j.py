import yaml
import json
import re
import ast
import pandas as pd
from tqdm import tqdm

from neo4j import GraphDatabase
from haystack.document_stores import ElasticsearchDocumentStore


class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd

        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.driver.session(
                database=db) if db is not None else self.driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


def merge_node(node_labels, node_attributes, db=None):
    with conn.driver.session(database=db) if db is not None else conn.driver.session() as session:
        node_labels = ":".join(node_labels)
        node_attributes = "{"+", ".join([re.sub('[^A-Za-z0-9]+', '_', k)+" : '"+str(node_attributes[k]).replace(
            "'", "").encode("ascii", "ignore").decode()+"'" for k in node_attributes.keys() if not k[0].isdigit()])+"}"
        # print("MERGE (p:{} {}) RETURN p".format(node_label, node_attributes))'
        print("MERGE (p:{} {}) RETURN p".format(node_labels, node_attributes))
        print("\n")
        return session.run("MERGE (p:{} {}) RETURN p".format(node_labels, node_attributes)).single().value()


def create_index(node_index_name, node_label, node_attribute, db=None):
    with conn.driver.session(database=db) if db is not None else conn.driver.session() as session:
        # .single().value()
        return session.run("CREATE INDEX {} IF NOT EXISTS FOR (n:{}) ON (n.{})".format(node_index_name, node_label, node_attribute))


def merge_edge(source_node_label, source_node_attribute, target_node_label, target_node_attribute, relation_type, edge_attributes, db=None):

    with conn.driver.session(database=db) if db is not None else conn.driver.session() as session:
        source_attributes = "{"+", ".join([k+" : '"+str(source_node_attribute[k]).replace(
            "'", "").encode("ascii", "ignore").decode()+"'" for k in source_node_attribute.keys()])+"}"
        target_attributes = "{"+", ".join([k+" : '"+str(target_node_attribute[k]).replace(
            "'", "").encode("ascii", "ignore").decode()+"'" for k in target_node_attribute.keys()])+"}"
        edge_attributes = "{"+", ".join(
            [k+" : '"+edge_attributes[k]+"'" for k in edge_attributes.keys()])+"}"
        # .single().value()
        return session.run("MATCH (s:{} {}), (t:{} {}) MERGE (s)<-[e:{} {}]-(t) RETURN e".format(source_node_label, source_attributes, target_node_label, target_attributes, relation_type, edge_attributes))


def _generate_nodes(node_df, db=None):
    for idx, node in node_df.iterrows():
        node_labels = ['Entity']
        print(node.metadata)
        node_attributes = ast.literal_eval(node.metadata)
        node_attributes['doc_id'] = str(node.doc_id)
#       for k,v in node_attributes.items():
#         print(k,": ",v)
#       print("\n")
        merge_node(node_labels, node_attributes, db)

    create_index("entity_id_index", "Entity", "wikiPageID", db)


def _generate_edges(entities_triples_df, relations_dict, db=None):
    for idx, triple in tqdm(entities_triples_df.iterrows(), total=len(entities_triples_df)):
        # Can just use the universal node label since node_id already uniquely identifies the node
        source_node_label = "Entity"
        source_node_attributes = {
            "entity": triple.subject, "doc_id": str(triple.doc_id)}
        target_node_label = "Entity"
        target_node_attributes = {
            "entity": triple.object, "doc_id": str(triple.doc_id)}
        relation_type = triple.relation.replace(" ", "_")
        relation_type = re.sub('[^A-Za-z0-9]+', '_', relation_type)
        edge_attributes = {"relation_id": str(
            relations_dict[triple.relation]), "doc_id": str(triple.doc_id), "timestamp": str(triple.timestamp)}
        merge_edge(source_node_label, source_node_attributes, target_node_label,
                   target_node_attributes, relation_type, edge_attributes, db)


if __name__ == '__main__':

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    conn = Neo4jConnection(uri=config['neo4j']['uri'],
                           user=config['neo4j']['user'],
                           pwd=config['neo4j']['pwd'])

    conn.query('CREATE CONSTRAINT {}s IF NOT EXISTS ON (x:{})     ASSERT x.id IS UNIQUE'.format(
        "entity", "Entity"), db=config['neo4j']['db'])

    nodes_dataframe = pd.read_csv(config['data']['entities_data_path'])
    _generate_nodes(nodes_dataframe, db=config['neo4j']['db'])

    relations_dict_file = open(config['graph']['relations_dict_path'])
    relations_dict = json.load(relations_dict_file)
    relations_dataframe = pd.read_csv(config['data']['relations_data_path'])
    print(relations_dataframe.info())
    relations_dataframe.drop_duplicates(inplace=True)
    print(relations_dataframe.info())

    document_store = ElasticsearchDocumentStore(host="elasticsearch",
                                                port="9200",
                                                username="elastic",
                                                password="changeme",
                                                scheme="https",
                                                verify_certs=False,
                                                index='documents_m2e2',
                                                search_fields=['content', 'title'])

    documents = document_store.get_all_documents()

    articles_df = pd.DataFrame(
        columns=['doc_id', 'timestamp', 'elasticsearch_ID'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'],
                               document.meta['timestamp'], document.id]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()

    print(articles_df.head())

    merged_df = pd.merge(
        relations_dataframe, articles_df, on=['doc_id'])
    print(merged_df.info())
    _generate_edges(merged_df, relations_dict)
