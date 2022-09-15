# importing dependencies
import streamlit as st
import numpy as np
import os
import requests
import re
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
from collections import Counter
from collections import defaultdict
from elasticsearch import Elasticsearch

import folium
from folium import Marker
from folium.plugins import MarkerCluster
import datetime


ELASTIC_URL = os.environ['ELASTICSEARCH_HOST_PORT']
ELASTIC_USERNAME = os.environ['ELASTIC_USERNAME']
ELASTIC_PASSWORD = os.environ['ELASTIC_PASSWORD']

es = Elasticsearch(ELASTIC_URL,
                   # ca_certs="",
                   basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
                   verify_certs=False)


class Report:
    def __init__(self, id, title="", body="", date="", text_entities=[], images={}, visual_entities=[], geo_data=[], timestamp=""):
        self.id = id
        self.title = title
        self.body = body
        self.date = date
        self.text_entities = text_entities
        self.images = images
        self.visual_entities = visual_entities
        self.geo_data = geo_data
        self.timestamp = timestamp

    def __str__(self):
        return f'{self.id}: {self.title}'

    def __repr__(self):
        return str(self)


class Entity:
    def __init__(self, id, title="", body="", details={}, associated_entities=[], media=[], associated_reports=[]):
        self.id = id
        self.title = title
        self.body = body
        self.details = details
        self.associated_entities = associated_entities
        self.media = media
        self.associated_reports = associated_reports

    def __str__(self):
        return f'{self.id}: {self.title}'

    def __repr__(self):
        return str(self)


st.set_page_config(
    page_title="Multimedia Information Retrival",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# For navigation using query params
query_params: Dict = st.experimental_get_query_params()

# Initialise state variable
st.session_state['search'] = st.session_state['searchbar'] if 'searchbar' in st.session_state else query_params.get('search', [
                                                                                                                    ''])[0]
st.session_state['entity'] = query_params.get('entity', [''])[0]
st.session_state['report'] = query_params.get('report', [''])[0]
st.session_state['reports'] = st.session_state['reports'] if 'reports' in st.session_state else None


def escape_latex(text: str) -> str:
    return text.replace('$', '\$')


def generate_hypertext(text_entities: List[Dict], body: str):
    hypertext = body
    # Sort entities such that the longer mentions get replaced first
    sorted_entities = sorted(
        text_entities, key=lambda entity: len(entity['mention']), reverse=True)
    for entity in sorted_entities:
        if entity["mention"] != " " and entity['entity_link'] != "Unknown":
            # Workaround for closing in entity mention
            mention = entity['mention'].replace(')', '\)')
            print(hypertext, mention)
            hypertext = re.sub(
                fr"\b{mention}\b", f"<a href='?entity={entity['entity_link']}' target='_self'>{entity['mention']}</a>", hypertext)
    return hypertext


def get_image(server_path: str):
    body = {'server_path': server_path}
    res = requests.get(
        'http://image_server:8000/download/', json=body)
    return res.json()['image']


@ st.experimental_memo(show_spinner=False)
def get_entity_name(id: int) -> str:
    if id == "-1":
        return "Unknown"
    else:
        res = es.search(index="wikipedia", query={"term": {"_id": id}})
    return res['hits']['hits'][0]['_source']['title']


@ st.experimental_memo(show_spinner=False)
def search_reports(query: str, start_date=None, end_date=None) -> List[Report]:
    es_query = {"bool": {
        "filter": {"match_all": {}},
        "must": [{"match": {"content": query}}]}
    }
    if start_date or end_date:
        es_query["bool"]["filter"] = {
            "range": {"timestamp": {"gte": start_date, "lte": end_date}}}

    if query == "*":
        res = es.search(index="documents", query={"match_all": {}}, size=10000)
    else:
        res = es.search(index="documents", query=es_query, size=20)
    results = []
    for doc in res['hits']['hits']:
        id = doc['_source']['ID']
        title = doc['_id']
        body = doc['_source']['content']
        geo_data = doc['_source']['geo_data']
        timestamp = doc['_source']['timestamp']
        text_entities = doc['_source']['text_entities']
        visual_entities = res['hits']['hits'][0]['_source']['visual_entities'] if res['hits']['hits'][0]['_source']['images'] else []
        result = Report(id=id, title=title, body=body,
                        geo_data=geo_data, timestamp=timestamp, text_entities=text_entities, visual_entities=visual_entities)
        results.append(result)
    return results


@ st.experimental_memo(show_spinner=False)
def get_report(id: int) -> Report:
    # title = "Hamilton claims Vettel 'incomparable' to other F1 stars"
    # body = """
    # <b>Lewis Hamilton has described Sebastian Vettel as being 'unlike any driver' from F1 past or present after the German announced his retirement from the sport.</b>
    # <br>
    # The two drivers joined the grid full-time within half a year of one another and until recently were fierce rivals on the track. But as the distance separating the pair on the grid has grown, more common ground has been discovered with Hamilton describing the four-time champion as a 'powerful ally' in his battle against racism and social injustice. Vettel has backed numerous environmental causes this year, including wearing a shirt in Miami warning of the event would be the "first underwater grand prix" if climate change is not addressed now.  “Over time, we have started to see one another taking those brave steps and standing out for things that we believe in and have been able to support each other," said Hamilton.  "He has been so supportive to me and I like to think I have supported him also and have come to realise that we have a lot more in common than just the driving passion.  “So it is really me and him that have been stepping out into the uncomfortable light and trying to do something with the platform that we have.  “And that is for me why he is very much unlike any of the drivers that have been here past and present.” Vettel a "great all-round competitor"  Vettel finished in the top five of the drivers' championship for 11 consecutive seasons from 2009 to 2019 but has since failed to place in the top 10.  Asked how the pair's relationship has evolved over the years, Hamilton said: “The racing part of things, he was just incredibly quick.  "He was very, very intelligent, a very good engineer, I think, and just very, very precise on track.  “He was just a great all-around competitor, very fair but also very strong and firm on track.  “He has never been someone to blame other people for mistakes, he would always put his hand up and say it was his fault which I always thought was honourable.  “Naturally, when you are focused on winning championships and stuff, when we were younger, we didn’t have time to stop and talk about what we do in our own personal lives and the things that we cared about."
    # """
    res = es.search(index="documents", query={"term": {"ID": id}})
    body = res['hits']['hits'][0]['_source']['content']
    text_entities = res['hits']['hits'][0]['_source']['text_entities']
    hypertext = generate_hypertext(text_entities, body)
    images = {}
    for server_path in res['hits']['hits'][0]['_source']['images']:
        images[server_path] = get_image(server_path)
    visual_entities = res['hits']['hits'][0]['_source']['visual_entities'] if images else [
    ]
    result = Report(id=res['hits']['hits'][0]['_source']['ID'],
                    title=res['hits']['hits'][0]['_id'], body=hypertext, images=images, visual_entities=visual_entities)
    return result


def search_entities(query: str) -> List[Entity]:
    res = es.search(index="wikipedia", query={"match": {"title": query}})
    results = []
    for doc in res['hits']['hits']:
        id = doc['_id']
        title = doc['_source']['title']
        body = doc['_source']['content']
        result = Entity(id=id, title=title, body=body)
        results.append(result)
    return results


def get_entity(id: int) -> Entity:
    # results = Entity(id=1, title="Lewis Hamilton", details={'Last Spotted': 'Kenya (17 Aug 2022)', 'Location': 'Pokot District of the eastern Karamoja region in Kenya', 'Activities': ['Lewis Hamilton, 37, enjoys relaxing trip to Rwanda during F1 summer break', 'Hamilton came second to Max Verstappen in the recent Hungarian Grand Prix']}, associated_entities=[Entity(id=2, title="Max Verstappen")], media=['/images/lewis-1.jpg', '/images/lewis-2.jpg', '/images/lewis-3.jpg'], associated_reports=[Report(id=1, title="Hamilton claims Vettel 'incomparable' to other F1 stars", date="23 Dec 2020"), Report(id=2, title="Lewis Hamilton explains why he turned Tom Cruise down", date="20 Dec 2020"), Report(id=3, title="Lewis Hamilton: Sebastian Vettel has made F1 feel less lonely", date="09 Dec 2020")])
    res = es.search(index="wikipedia", query={"term": {"_id": id}})
    associated_reports = search_reports(
        res['hits']['hits'][0]['_source']['title'])
    result = Entity(id=res['hits']['hits'][0]['_id'],
                    title=res['hits']['hits'][0]['_source']['title'], body=res['hits']['hits'][0]['_source']['content'], associated_reports=associated_reports)
    return result


# Info
st.info('Elasticsearch ID temporarily displayed as report title until title is populated to ES')

# Search Bar
inp = st.text_input(label='Search', key='searchbar',
                    value=st.session_state['search'], placeholder='Lewis Hamilton')
reports = None

if inp != "":
    # Reset entity and report parameters on new query
    st.session_state['entity'] = ""
    st.session_state['report'] = ""
    query_params["search"] = inp
    st.experimental_set_query_params(**query_params)
    with st.spinner(text=f'Searching: {inp}'):
        reports: List[Report] = search_reports(inp)
        entities: List[Entity] = search_entities(inp)
        st.session_state['reports'] = reports
elif inp != query_params.get('search', [''])[0]:
    # User clears input
    query_params["search"] = inp
    st.experimental_set_query_params(**query_params)


# TODO: Navigate using states instead of href to preserve (reports) cache and prevent reloading (+ requerying) when back button is used.
if st.session_state['search']:
    reports_tab, maps_tab = st.tabs(["📄 Reports", "📍 Maps"])

    with reports_tab:
        col1, col2 = st.columns([8, 2])
        with col1:
            if reports:
                st.header("Reports")
                st.success(f"We found {len(reports)} matches")
                for doc in reports:
                    st.subheader(
                        f"<a href='?report={doc.id}' target='_self'>{doc.title}</a>", anchor="")
                    st.write(escape_latex(doc.body))

        with col2:
            if entities:
                st.header("Entity Results")
                st.success(f"We found {len(entities)} matches")
                for entity in entities:
                    st.subheader(
                        f"<a href='?entity={entity.id}' target='_self'>{entity.title}</a>", anchor="")

    with maps_tab:
        def extract_entities(reports: List[Report]) -> Tuple[Counter, defaultdict]:
            entities = Counter()
            geo_entities = defaultdict(list)
            for report in reports:
                text_entities = [entity['entity_link']
                                 for entity in report.text_entities if entity['entity_link'] != -1]
                entities.update(text_entities)
                visual_entities = [entity['person_id']
                                   for entity in report.visual_entities]
                visual_entities = [
                    person_id for person_ids in visual_entities for person_id in person_ids if person_id != -1]
                entities.update(visual_entities)
                for location in report.geo_data:
                    geo_entities[(location['entity_name'], location['latitude'], location['longitude'])].append(
                        {"ID": report.id, "timestamp": report.timestamp, "entities": set(text_entities+visual_entities)})
            return entities, geo_entities

        def display_map(markers, checkbox):
            m = folium.Map([0, 0], tiles='OpenStreetMap', zoom_start=3)
            # Add marker for Location
            for location in markers.keys():
                location_name = location[0]
                location_latitude = location[1]
                location_longitude = location[2]
                entities = [report['entities'] for report in markers[location]]
                entities_set = set.union(*entities)
                for entity in entities_set:
                    # display the marker as long as one of its entity is selected
                    if checkbox[entity]:
                        folium.Marker(location=[location_latitude, location_longitude],
                                      popup=folium.Popup(f"""
                                    <b>{location_name}</b>
                                    <br><br>
                                    {'<br>'.join([f"{report['timestamp']}: <b><a href='http://localhost:8501/?report={report['ID']}' target='_blank'>{report['ID']}</a></b>" for report in markers[location]])}<br>
                                    """, min_width=180, max_width=180),
                                      icon=folium.Icon()).add_to(m)

            return st.markdown(m._repr_html_(), unsafe_allow_html=True)

        # def generate_markers(reports: List[Report]):
        #     markers = defaultdict(list)
        #     # Sort reports by descending timestamp
        #     sorted_reports = sorted(
        #         reports, key=lambda report: report.timestamp, reverse=True)
        #     for report in sorted_reports:
        #         for geo in report.geo_data:
        #             markers[(geo["entity_name"], geo["latitude"],
        #                     geo["longitude"])].append(f"{report.timestamp}: <b><a href='http://localhost:8501/?report={report.id}' target='_blank'>{report.id}</a></b>")  # Hard code URL
        #     return markers

        start_date = st.date_input(
            "Start Date", max_value=datetime.date.today(), value=datetime.date(1970, 1, 1))
        end_date = st.date_input(
            "End Date", max_value=datetime.date.today())
        reports: List[Report] = search_reports(
            inp, start_date=start_date, end_date=end_date)
        entities, markers = extract_entities(reports)
        checkbox = {entity_id: False for entity_id in dict(entities)}

        col1, col2 = st.columns([8, 2])

        with col2:
            for entity in entities.most_common():
                checked = st.checkbox(
                    label=f"{get_entity_name(entity[0])} ({entity[1]})", key=entity[0], value=checkbox[entity[0]])
                if checked:
                    checkbox[entity[0]] = True
                else:
                    checkbox[entity[0]] = False

        with col1:
            # markers = generate_markers(reports)
            display_map(markers, checkbox)

elif st.session_state['entity']:
    col1, col2 = st.columns([8, 2])

    entity_id = st.session_state['entity']
    with st.spinner(text=f"Loading Report {entity_id}"):
        entity: Entity = get_entity(entity_id)

    with col1:
        if entity:
            st.header(f"**Entity: {entity.title}**")
            st.write(
                f"<div style='text-align: justify;'>{entity.body}</div>", unsafe_allow_html=True)
            st.markdown("""---""")
            st.markdown(
                f"**Last Spotted**: {entity.details['Last Spotted'] if 'Last Spotted' in entity.details else ''}")
            st.markdown(
                f"**Location**: {entity.details['Location'] if 'Location' in entity.details else ''}")
            st.markdown(f"**Activities:**")
            for activity in entity.details['Activities'] if 'Activities' in entity.details else '':
                st.markdown(f"{activity}")
            st.markdown(f"**Associated Entities**:")
            for associated_entity in entity.associated_entities:
                st.markdown(
                    f"<a href='?entity={associated_entity.id}' target='_self'>{associated_entity.title}</a>", unsafe_allow_html=True)
            st.markdown(f"**Media**:")

            # For future development: https://github.com/vivien000/st-clickable-images
            for image_path in entity.media:
                im = Image.open(image_path)
                im.thumbnail((256, 256))
                st.image(im)

    with col2:
        if entity:
            st.header("Reports")
            for report in entity.associated_reports:
                st.subheader(
                    f"<a href='?report={report.id}' target='_self'>{report.title}</a>", anchor="")
                st.write(report.date)

elif st.session_state['report']:
    col1, col2 = st.columns([8, 2])

    report_id = st.session_state['report']
    with st.spinner(text=f"Loading Report {report_id}"):
        report: Report = get_report(report_id)

    with col1:
        if report:
            st.header(f"**Report: {report.title}**")
            # NOTE: Unsafe
            st.write(
                f"<div style='text-align: justify;'>{report.body}</div>", unsafe_allow_html=True)
            st.markdown("""---""")

            for server_path, image in report.images.items():
                im = Image.fromarray(np.asarray(image).astype(np.uint8))
                for visual_entity in report.visual_entities:
                    if visual_entity["file_name"] != server_path:
                        continue
                    else:
                        # Generate checkbox
                        for person_id in set(visual_entity['person_id']):
                            st.checkbox(
                                label=f"{get_entity_name(person_id)}", key=f"{server_path}_{person_id}", value=True)

                        # Generate bounding box
                        for person_idx, bbox in enumerate(visual_entity['person_bbox']):
                            draw = ImageDraw.Draw(im)
                            if st.session_state[f"{server_path}_{visual_entity['person_id'][person_idx]}"]:
                                draw.rectangle(
                                    bbox)
                                # Top left corner
                                draw.text((bbox[0], bbox[1]),
                                          f"{get_entity_name(visual_entity['person_id'][person_idx])}, Conf: {visual_entity['person_conf'][person_idx]}")
                        break

                # im.thumbnail((256, 256))
                st.image(im)
                st.markdown("***")  # Line Break

    with col2:
        # For future development
        pass

else:
    pass
