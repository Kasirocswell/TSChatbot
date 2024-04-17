import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from HTMLTemplate import css, user_template, bot_template
from langchain_community.document_loaders import SeleniumURLLoader
import os
from dotenv import load_dotenv
embedding_model_name=os.environ.get('EMBEDDING_MODEL_NAME')

urls = [
    'https://teachstone.com/',
    'https://teachstone.com/support/',
    'https://teachstone.com/class/',
    'https://teachstone.com/focus/',
    'https://teachstone.com/measure/',
    'https://teachstone.com/improve/',
    'https://teachstone.com/for-teachers/',
    'https://teachstone.com/for-coaches/',
    'https://teachstone.com/for-observers/',
    'https://teachstone.com/leaders/',
    'https://teachstone.com/for-class-affiliates/',
    'https://teachstone.com/research/',
    'https://teachstone.com/class/headstart/',
    'https://teachstone.com/products-and-trainings/',
    'https://teachstone.com/getting-started-with-class/',
    'https://teachstone.com/class-foundations-for-teachers/',
    'https://teachstone.com/a-class-overview-for-leaders/',
    'https://teachstone.com/a-class-primer-for-teachers/',
    'https://teachstone.com/class-2nd-educator-overview/',
    'https://teachstone.com/intro-to-interactions-for-educators/',
    'https://teachstone.com/class-environment/',
    'https://teachstone.com/class-observation-training/',
    'https://teachstone.com/class-train-the-trainer/',
    'https://teachstone.com/class-2nd-edition-trainer-transition-training/',
    'https://teachstone.com/class-2nd-measurement-trainer/',
    'https://teachstone.com/class-observer-supports/',
    'https://teachstone.com/coding-support/',
    'https://teachstone.com/class-observation-supports/',
    'https://teachstone.com/class-2nd-observer-transition-training/',
    'https://teachstone.com/program-services-support/',
    'https://teachstone.com/myteachstone/',
    'https://teachstone.com/class-observation-services/',
    'https://teachstone.com/custom-events/',
    'https://teachstone.com/meaningful-interactions-at-home/',
    'https://teachstone.com/coaching-training-and-certification/',
    'https://teachstone.com/1-1-video-coaching/',
    'https://teachstone.com/class-group-coaching/',
    'https://teachstone.com/class-master-coaching/',
    'https://teachstone.com/coach-feedback/',
    'https://teachstone.com/professional-development-for-educators/',
    'https://teachstone.com/cda-programs/',
    'https://teachstone.com/banking-time/',
    'https://teachstone.com/instructional-support-essentials-for-teachers/',
    'https://teachstone.com/interactions-at-the-heart-of-healing/',
    'https://teachstone.com/thinking-thriving/',
    'https://teachstone.com/class-supplies/',
    'https://teachstone.com/class-support-kits/',
    'https://teachstone.com/resources/',
    'https://teachstone.com/podcasts/',
    'https://teachstone.com/upcoming-events/',
    'https://teachstone.com/about-teachstone/',
    'https://teachstone.com/our-impact/',
    'https://teachstone.com/our-partners/',
    'https://teachstone.com/careers/',
    'https://teachstone.com/certified-b-corporation/',
    'https://teachstone.com/class-2nd-edition/',
    'https://teachstone.com/regional-trainings/',
    'https://teachstone.com/overcome-workforce-challenges/',
    'https://teachstone.com/bridging-the-literacy-gap/',
    'https://teachstone.com/decrease-challening-behaviors/',
    'https://teachstone.com/interact-class-summit/',
    'https://teachstone.com/class-en-francais/',
    'https://teachstone.com/collecion-de-productos-en-esp/',
    'https://teachstone.com/privacy-policy/',
    'https://teachstone.com/terms-conditions/',
    'https://teachstone.com/web-accessibility/',
    'https://teachstone.com/questions/your-account/',
    'https://teachstone.com/support/login/',
    'https://teachstone.com/support/password-settings/',
    'https://teachstone.com/support/email-settings/',
    'https://teachstone.com/questions/testing-certifications/',
    'https://teachstone.com/support/guide-to-purchasing-recertification/',
    'https://teachstone.com/support/observers/',
    'https://teachstone.com/support/trainers/',
    'https://teachstone.com/support/continuing-education-units-ceu/',
    'https://teachstone.com/support/child-development-association/',
    'https://teachstone.com/support/online-services/',
    'https://teachstone.com/support/certification-resources/',
    'https://teachstone.com/support/customer-advocate-program/',
    'https://teachstone.com/support/myteachstone/',
    'https://teachstone.com/support/class-access/',
    'https://teachstone.com/support/technology/',
    'https://teachstone.com/support/facilitators/',
    'https://teachstone.com/support/coaches/',
    'https://teachstone.com/support/observers-trainees/',
    'https://teachstone.com/support/program-staff/',
    'https://teachstone.com/support/our-impact/',
    'https://teachstone.com/support/coding-support/',
    'https://teachstone.com/support/interactions-data-training/',
    'https://teachstone.com/support/observations/',
    'https://teachstone.com/support/quality-assurance/',
    'https://teachstone.com/support/events/',
    'https://teachstone.com/support/podcasts/',
    'https://teachstone.com/support/',
    'https://teachstone.com/store/',
    'https://teachstone.com/store/education-resources/',
    'https://teachstone.com/store/class-supplies/',
    'https://teachstone.com/store/class-support-kits/',
    'https://teachstone.com/store/literacy-resources/',
    'https://teachstone.com/store/instructional-supports/',
    'https://teachstone.com/store/behavior-management/',
    'https://teachstone.com/store/mentor-coach-supports/',
    'https://teachstone.com/store/classroom-management/',
    'https://teachstone.com/store/social-emotional-learning-resources/',
    'https://teachstone.com/store/home-learning-resources/',
    'https://teachstone.com/store/family-engagement/',
    'https://teachstone.com/store/language-development/',
    'https://teachstone.com/store/',
    'https://teachstone.com/store/education-resources/',
    'https://teachstone.com/store/class-supplies/',
    'https://teachstone.com/store/class-support-kits/',
    'https://teachstone.com/store/literacy-resources/',
    'https://teachstone.com/store/instructional-supports/',
    'https://teachstone.com/store/behavior-management/',
    'https://teachstone.com/store/mentor-coach-supports/',
    'https://teachstone.com/store/classroom-management/',
    'https://teachstone.com/store/social-emotional-learning-resources/',
    'https://teachstone.com/store/home-learning-resources/',
    'https://teachstone.com/store/family-engagement/',
    'https://teachstone.com/store/language-development/',
    'https://teachstone.com/store/class-supplies/',
    'https://teachstone.com/store/class-support-kits/',
    'https://teachstone.com/store/',
    'https://teachstone.com/store/education-resources/',
    'https://teachstone.com/store/literacy-resources/',
    'https://teachstone.com/store/instructional-supports/',
    'https://teachstone.com/store/behavior-management/',
    'https://teachstone.com/store/mentor-coach-supports/',
    'https://teachstone.com/store/classroom-management/',
    'https://teachstone.com/store/social-emotional-learning-resources/',
    'https://teachstone.com/store/home-learning-resources/',
    'https://teachstone.com/store/family-engagement/',
    'https://teachstone.com/store/language-development/',
    'https://teachstone.com/store/literacy-resources/',
    'https://teachstone.com/store/instructional-supports/',
    'https://teachstone.com/store/behavior-management/',
    'https://teachstone.com/store/mentor-coach-supports/',
    'https://teachstone.com/store/classroom-management/',
    'https://teachstone.com/store/social-emotional-learning-resources/',
    'https://teachstone.com/store/home-learning-resources/',
    'https://teachstone.com/store/family-engagement/',
    'https://teachstone.com/store/language-development/',
    'https://store.teachstone.com/',
    'https://store.teachstone.com/all-products/',
    'https://store.teachstone.com/face-to-face/',
    'https://store.teachstone.com/?setCurrencyId=1',
    'https://store.teachstone.com/?setCurrencyId=2',
    'https://store.teachstone.com/observer-recertification-new/',
    'https://store.teachstone.com/class-overview-for-leaders/',
    'https://store.teachstone.com/a-class-primer-for-teachers-online-course/',
    'https://store.teachstone.com/class-emotional-support-kit-recognizing-and-managing-feelings-pre-k/',
    'https://store.teachstone.com/class-strategy-cards/',
    'https://store.teachstone.com/class-2nd-edition-intro-to-interactions-for-educators/',
    'https://store.teachstone.com/instructional-support-kit-for-teachers/',
    'https://store.teachstone.com/class-foundations-for-teachers/',
    'https://store.teachstone.com/coaching-with-myteachstone/',
    'https://store.teachstone.com/class-dimensions-guide/',
    'https://store.teachstone.com/score-sheets/',
    'https://store.teachstone.com/class-manual/',
    'https://store.teachstone.com/trainer-recertification-new/',
    'https://store.teachstone.com/class-dictionary/',
    'https://store.teachstone.com/class-dimensions-overview/',
    'https://store.teachstone.com/teacher/',
    'https://store.teachstone.com/class-observation-training/',
    'https://store.teachstone.com/all-products/?setCurrencyId=1',
    'https://store.teachstone.com/all-products/?setCurrencyId=2',
    'https://store.teachstone.com/class-observation-support-reducing-bias/',
    'https://store.teachstone.com/class-observation-support-settings-serving-children-with-disabilities/',
    'https://store.teachstone.com/class-2nd-edition-pre-k-3-manual-options/',
    'https://store.teachstone.com/class-observation-support-settings-with-dual-language-learners-1/',
    'https://store.teachstone.com/infant-and-toddler-class-social-and-emotional-development-kit-regulation/',
    'https://store.teachstone.com/infant-and-toddler-class-cognitive-support-kit-cause-and-effect/',
    'https://store.teachstone.com/class-dictionary-class-strategy-cards-bundle/',
    'https://store.teachstone.com/video-library-subscription/',
    'https://store.teachstone.com/interactions-at-the-heart-of-healing-trauma-informed-professional-development-series/',
    'https://store.teachstone.com/two-week-extension/',
    'https://store.teachstone.com/instructional-support-essentials-for-teachers/',
    'https://store.teachstone.com/a-class-primer-for-teachers/',
    'https://store.teachstone.com/myteachstone/',
    'https://store.teachstone.com/class-discussion-toolkit-1/',
    'https://store.teachstone.com/class-train-the-trainer/',
    'https://store.teachstone.com/class-feedback-strategies/',
    'https://store.teachstone.com/introduction-to-the-class-tool-training/',
    'https://store.teachstone.com/instructional-support-strategies-1/',
    'https://store.teachstone.com/class-group-coaching-mmci/',
    'https://store.teachstone.com/class-1-on-1-video-coaching-mtp/',
    'https://store.teachstone.com/class-video-library-companion/',
    'https://store.teachstone.com/face-to-face/?setCurrencyId=1',
    'https://store.teachstone.com/face-to-face/?setCurrencyId=2',
    'https://store.teachstone.com/product-type/',
    'https://store.teachstone.com/observer-recertification-new/?setCurrencyId=1',
    'https://store.teachstone.com/observer-recertification-new/?setCurrencyId=2',
    'https://store.teachstone.com/online-programs/',
    'https://store.teachstone.com/cda-facilitated/',
    'https://store.teachstone.com/cda-on-demand/',
    'https://store.teachstone.com/class-overview-for-leaders/?setCurrencyId=1',
    'https://store.teachstone.com/class-overview-for-leaders/?setCurrencyId=2',
    'https://store.teachstone.com/class-2008-resources-for-class-2nd-edition-observers/',
    'https://store.teachstone.com/a-class-primer-for-teachers-online-course/?setCurrencyId=1',
    'https://store.teachstone.com/a-class-primer-for-teachers-online-course/?setCurrencyId=2',
    'https://store.teachstone.com/class-foundations-for-teachers-facilitation-supplement-10-00-55-00/',
    'https://store.teachstone.com/banking-time-investing-in-relationships/',
    'https://store.teachstone.com/class-emotional-support-kit-recognizing-and-managing-feelings-pre-k/?setCurrencyId=1',
    'https://store.teachstone.com/class-emotional-support-kit-recognizing-and-managing-feelings-pre-k/?setCurrencyId=2',
    'https://store.teachstone.com/toolkits/',
    'https://store.teachstone.com/class-literacy-support-kit-pre-k-k-45-00-385-00/',
    'https://store.teachstone.com/pre-k-class-observation-trainer-version-4-upgrade/',
    'https://store.teachstone.com/class-strategy-cards/?setCurrencyId=1',
    'https://store.teachstone.com/class-strategy-cards/?setCurrencyId=2',
    'https://store.teachstone.com/print/',
    "https://store.teachstone.com/online/",
    "https://store.teachstone.com/coaching-with-myteachstone/?setCurrencyId=1",
    "https://store.teachstone.com/coaching-with-myteachstone/?setCurrencyId=2",
    "https://store.teachstone.com/mmci-renewal/",
    "https://store.teachstone.com/class-dimensions-guide/?setCurrencyId=1",
    "https://store.teachstone.com/class-dimensions-guide/?setCurrencyId=2",
    "https://store.teachstone.com/score-sheets/?setCurrencyId=1",
    "https://store.teachstone.com/score-sheets/?setCurrencyId=2",
    "https://store.teachstone.com/class-manual/?setCurrencyId=1",
    "https://store.teachstone.com/class-manual/?setCurrencyId=2",
    "https://store.teachstone.com/trainer-recertification-new/?setCurrencyId=1",
    "https://store.teachstone.com/trainer-recertification-new/?setCurrencyId=2",
    "https://store.teachstone.com/recertification-des-observateurs-de-class-en-francais/",
    "https://store.teachstone.com/class-dictionary/?setCurrencyId=1",
    "https://store.teachstone.com/class-dictionary/?setCurrencyId=2",
    "https://store.teachstone.com/your-role/",
    "https://store.teachstone.com/class-dimensions-overview/?setCurrencyId=1",
    "https://store.teachstone.com/class-dimensions-overview/?setCurrencyId=2",
    "https://store.teachstone.com/teacher/?setCurrencyId=1",
    "https://store.teachstone.com/teacher/?setCurrencyId=2",
    "https://store.teachstone.com/class-observation-training/?setCurrencyId=1",
    "https://store.teachstone.com/class-observation-training/?setCurrencyId=2",
    "https://store.teachstone.com/age-levels/",
    "https://store.teachstone.com/infant-toddler/",
    "https://store.teachstone.com/class-observation-support-reducing-bias/?setCurrencyId=1",
    "https://store.teachstone.com/class-observation-support-reducing-bias/?setCurrencyId=2",
    "https://store.teachstone.com/class-observation-support-settings-serving-children-with-disabilities/?setCurrencyId=1",
    "https://store.teachstone.com/class-observation-support-settings-serving-children-with-disabilities/?setCurrencyId=2",
    "https://store.teachstone.com/class-2nd-edition-pre-k-3-manual-options/?setCurrencyId=1",
    "https://store.teachstone.com/class-2nd-edition-pre-k-3-manual-options/?setCurrencyId=2",
    "https://store.teachstone.com/class-observation-support-settings-with-dual-language-learners-1/?setCurrencyId=1",
    "https://store.teachstone.com/class-observation-support-settings-with-dual-language-learners-1/?setCurrencyId=2",
    "https://store.teachstone.com/infant-and-toddler-class-social-and-emotional-development-kit-regulation/?setCurrencyId=1",
    "https://store.teachstone.com/infant-and-toddler-class-social-and-emotional-development-kit-regulation/?setCurrencyId=2",
    "https://store.teachstone.com/infant-and-toddler-class-cognitive-support-kit-cause-and-effect/?setCurrencyId=1",
    "https://store.teachstone.com/infant-and-toddler-class-cognitive-support-kit-cause-and-effect/?setCurrencyId=2",
    "https://store.teachstone.com/class-dictionary-class-strategy-cards-bundle/?setCurrencyId=1",
    "https://store.teachstone.com/class-dictionary-class-strategy-cards-bundle/?setCurrencyId=2",
    "https://store.teachstone.com/video-library-subscription/?setCurrencyId=1",
    "https://store.teachstone.com/video-library-subscription/?setCurrencyId=2",
    "https://store.teachstone.com/interactions-at-the-heart-of-healing-trauma-informed-professional-development-series/?setCurrencyId=1",
    "https://store.teachstone.com/interactions-at-the-heart-of-healing-trauma-informed-professional-development-series/?setCurrencyId=2",
    "https://store.teachstone.com/two-week-extension/?setCurrencyId=1",
    "https://store.teachstone.com/two-week-extension/?setCurrencyId=2",
    "https://store.teachstone.com/cda-primrose/",
    "https://store.teachstone.com/cda-advanced-special-needs/",
    "https://store.teachstone.com/cda-partners/",
    "https://store.teachstone.com/cda-infant-toddler/",
    "https://store.teachstone.com/cda-coaching/",
    "https://store.teachstone.com/cda-in-early-childhood/",
    "https://store.teachstone.com/cda-in-french/",
    "https://store.teachstone.com/cda-center-based-training-bundle/",
    "https://store.teachstone.com/cda-family-child-care-training-bundle/",
    "https://store.teachstone.com/cda-spanish/",
    "https://store.teachstone.com/cda-training-class-observation-tool/"
]

def load_urls(urls):
    loaders = SeleniumURLLoader(urls=urls)
    data = loaders.load()
    return data

def get_text_chunks(data):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks=text_splitter.split_documents(data)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    #embeddings=HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def process_data(urls):
    data = load_urls(urls)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()  # Consider using caching here as well if this is heavy
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 != 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot for our Own Website", page_icon=":chatbot:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chatbot For Teachstone 💬")

    if "conversation" not in st.session_state:
        vectorstore = process_data(urls)  # This will cache and only run once per session
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.chat_history = None

    user_question = st.text_input("Please ask a question")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("LLM Chatapp using Teachstone.com and store.Teachstone.com data")
        st.markdown('''
        This app is an LLM powered Chatbot built using:
        - Made with Streamlit
        - Using openAI LLM
        - With LangChain
        ''')

if __name__ == '__main__':
    main()
