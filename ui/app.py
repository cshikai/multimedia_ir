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
st.session_state['search'] =  st.session_state['searchbar'] if 'searchbar' in st.session_state else query_params.get('search', [''])[0]
st.session_state['entity'] =  query_params.get('entity', [''])[0]
st.session_state['report'] =  query_params.get('report', [''])[0]
st.session_state['documents'] = st.session_state['documents'] if 'documents' in st.session_state else None

# TODO: Connection to database
def get_reports(query: str) -> List[Document]:
    time.sleep(1)
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi pharetra, sem et suscipit lacinia, risus ante dictum ligula, pharetra aliquet erat purus eget metus. Nunc convallis lorem eu odio porta, a sollicitudin massa accumsan. Praesent sit amet mauris et nibh imperdiet scelerisque. Praesent nisi massa, ornare vel ornare in, sollicitudin et quam. Nunc mattis quam libero, nec convallis sem semper eget. Maecenas ultrices posuere lacus, nec volutpat erat suscipit id. Donec ut quam porttitor, porttitor neque ut, consequat justo. Integer fringilla, nisl sit amet congue maximus, nisi est ultrices arcu, non egestas risus magna ut dolor. Aenean ut vestibulum metus, sagittis feugiat quam. Curabitur iaculis libero ullamcorper diam efficitur varius. Phasellus ultrices massa vitae gravida gravida. In sed sollicitudin dolor. Sed finibus ligula at orci sagittis mollis. Proin id velit rhoncus, elementum odio ac, pharetra leo. Aenean ultrices leo et ultrices consectetur."
    results = [Document(id=1, title="Lorem Ipsum A", body=long_text), Document(id=2, title="Lorem Ipsum B", body=long_text), Document(id=3, title="Lorem Ipsum C", body=long_text)]
    return results

# TODO: Connection to database
def get_entities(query: str) -> List[Document]:
    time.sleep(1)
    results = [Entity(id=1, title="Entity A", body="A"), Entity(id=2, title="Entity B", body="B"), Entity(id=3, title="Entity C", body="C"), Entity(id=4, title="Entity D", body="D")]
    return results

# Search Bar
inp = st.text_input(label='Search', key='searchbar', value=st.session_state['search'], placeholder = 'Lewis Hamilton')
documents = None

if inp!= "":
    # Reset entity and report parameters on new query
    st.session_state['entity'] = ""
    st.session_state['report'] = ""
    query_params["search"]=inp
    st.experimental_set_query_params(**query_params)
    with st.spinner(text= f'Searching: {inp}'):
        documents: List[Document] = get_reports(inp)
        entities: List[Entity] = get_entities(inp)
        st.session_state['documents'] = documents
elif inp != st.session_state['search']:
    # User clears input 
    query_params["search"]=inp
    st.experimental_set_query_params(**query_params)


col1, col2 = st.columns([8, 2])

# TODO: Navigate using states instead of href to preserve cache and prevent reloading (+ requerying)) when back button is used
if st.session_state['search']:
    with col1:
        if documents:
            st.header("Reports")
            st.success(f"We found {len(documents)} matches")
            for doc in documents:
                st.subheader(f"<a href='?report={doc.id}' target='_self'>{doc.title}</a>", anchor="")
                st.write(doc.body)

    with col2:
        if entities:
            st.header("Entity Results")
            st.success(f"We found {len(entities)} matches")
            for entity in entities:
                st.subheader(f"<a href='?entity={entity.id}' target='_self'>{entity.title}</a>", anchor="")


# TODO: Add functions for retrieval
if st.session_state['entity']:
    with col1:
        st.subheader(f"**Entity: {st.session_state['entity']}**")
        st.markdown(f"**Last Spotted**: {'Singapore'}")
        st.markdown(f"**Location**: {'Singapore'}")
        st.markdown(f"**Activities: {'-'}**")
        st.markdown(f"**Associated Entities**:")
        st.markdown(f"<a href='?entity={8080}' target='_self'>Hamburger</a>", unsafe_allow_html=True)

    with col2:
        st.header("Reports")
        st.subheader(f"<a href='?report={1}' target='_self'>{'Document 1'}</a>", anchor="")
        st.write("23 Dec 2020")
        st.subheader(f"<a href='?report={2}' target='_self'>{'Document 2'}</a>", anchor="")
        st.write("20 Dec 2020")
        st.subheader(f"<a href='?report={3}' target='_self'>{'Document 3'}</a>", anchor="")
        st.write("09 Dec 2020")
        

# TODO: Add functions for retrieval
if st.session_state['report']:
    with col1:
        st.subheader(f"**Report: {st.session_state['report']}**")
        st.write('Text')

    with col2:
        # For future development
        pass