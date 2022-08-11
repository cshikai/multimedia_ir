#importing dependencies
import streamlit as st
import time
from typing import List, Dict

class Document:
    def __init__(self, id, title="", body=""):
        self.id = id
        self.title = title
        self.body = body

    def __str__(self):
        return f'{self.id}: {self.body}'

    def __repr__(self):
        return str(self)

class Entity:
    def __init__(self, id, title="", body=""):
        self.id = id
        self.title = title
        self.body = body

    def __str__(self):
        return f'{self.id}: {self.body}'

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
st.session_state['query'] =  query_params.get('search', [''])[0]
st.session_state['documents'] = st.session_state['documents'] if 'documents' in st.session_state else None

# TODO
def get_reports(query: str) -> List[Document]:
    time.sleep(1)
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi pharetra, sem et suscipit lacinia, risus ante dictum ligula, pharetra aliquet erat purus eget metus. Nunc convallis lorem eu odio porta, a sollicitudin massa accumsan. Praesent sit amet mauris et nibh imperdiet scelerisque. Praesent nisi massa, ornare vel ornare in, sollicitudin et quam. Nunc mattis quam libero, nec convallis sem semper eget. Maecenas ultrices posuere lacus, nec volutpat erat suscipit id. Donec ut quam porttitor, porttitor neque ut, consequat justo. Integer fringilla, nisl sit amet congue maximus, nisi est ultrices arcu, non egestas risus magna ut dolor. Aenean ut vestibulum metus, sagittis feugiat quam. Curabitur iaculis libero ullamcorper diam efficitur varius. Phasellus ultrices massa vitae gravida gravida. In sed sollicitudin dolor. Sed finibus ligula at orci sagittis mollis. Proin id velit rhoncus, elementum odio ac, pharetra leo. Aenean ultrices leo et ultrices consectetur."
    results = [Document(id=1, title="Lorem Ipsum A", body=long_text), Document(id=2, title="Lorem Ipsum B", body=long_text), Document(id=3, title="Lorem Ipsum C", body=long_text)]
    return results

# TODO
def get_entities(query: str) -> List[Document]:
    time.sleep(1)
    results = [Entity(id=1, title="Entity A", body="A"), Entity(id=2, title="Entity B", body="B"), Entity(id=3, title="Entity C", body="C"), Entity(id=4, title="Entity D", body="D")]
    return results


st.warning('Known Bug: You might need to input a value to the input widget twice as it resets to the default value when using session states (https://github.com/streamlit/streamlit/issues/5125)')

# Search Bar
inp = st.text_input(label='Search', key='searchbar', value=st.session_state['query'], placeholder = 'Lewis Hamilton')

# No reload if empty query
if(inp is None or inp == ""):
    documents = None
    entities = None
else:
    st.experimental_set_query_params(search=inp)
    with st.spinner(text= f'Searching: {inp}'):
        documents: List[Document] = get_reports(inp)
        entities: List[Entity] = get_entities(inp)
        st.session_state['documents'] = documents

col1, col2 = st.columns([8, 2])

with col1:
    if documents:
        st.header("Reports")
        st.success(f"We found {len(documents)} matches")
        for doc in documents:
            st.subheader(f'**{doc.title}**')
            st.write(doc.body)

with col2:
    if entities:
        st.header("Entity Results")
        st.success(f"We found {len(entities)} matches")
        for entity in entities:
            st.subheader(f'**{entity.title}**')