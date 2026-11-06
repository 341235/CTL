"""Local RAG demo."""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import streamlit as st

st.set_page_config(page_title='CTL · Governance RAG', layout='wide')
st.title('Governance RAG')
st.caption('Source passages · Publication dates · Retrieval comparison')

@st.cache_resource(show_spinner='Loading data and search indexes…')
def resources():
    from src.stage3_rag.pipeline import prepare_data, build_index
    from src.stage3_rag.generation import OllamaGenerator
    chunks, _, graph, entities, _ = prepare_data(ROOT)
    index = build_index(ROOT, chunks)
    mode = json.loads((ROOT / 'outputs/results/stage3/retrieval_manifest.json').read_text())['selected_text_mode']
    generator = OllamaGenerator(model='llama3.2:latest', cache_dir=ROOT / 'outputs/models/stage3/answers',
                                seed=42, temperature=0, max_new_tokens=192, num_ctx=4096)
    return index, graph, entities, mode, generator

with st.form('question_form'):
    question = st.text_area('Your question', 'What risks and safeguards does the dataset describe for AI agents?', height=90)
    variant = st.selectbox('Retrieval method', ['Text + Graph + Re-Ranking', 'Text + Graph', 'Text only'])
    submitted = st.form_submit_button('Ask question', type='primary')

if submitted:
    if not question.strip():
        st.info('Enter a question.')
    else:
        try:
            index, graph, entities, mode, generator = resources()
            use_graph = variant != 'Text only'
            with st.spinner('Retrieving sources and preparing the answer…'):
                hits = index.retrieve(question.strip(), mode=mode, top_k=4, candidate_k=20,
                    graph=graph if use_graph else None, entity_articles=entities if use_graph else None,
                    graph_weight=.35, rerank=variant == 'Text + Graph + Re-Ranking')
                result = generator.answer(question.strip(), hits)
            st.session_state['latest'] = {'question': question.strip(), **result}
        except Exception as error:
            st.error(f'The request could not be completed: {error}')

if 'latest' not in st.session_state:
    demo = ROOT / 'outputs/results/stage3/live_demo.json'
    if demo.exists():
        st.session_state['latest'] = json.loads(demo.read_text())

if 'latest' in st.session_state:
    result = st.session_state['latest']
    st.subheader('Answer')
    st.caption(result['question'])
    st.write(result.get('answer') or result.get('error', 'No answer'))
    status = result.get('status', 'unknown')
    st.caption(f'Validation status: {status} · Source IDs: {result.get("citations", [])}')
    if status != 'ok':
        st.warning(result.get('error', 'The answer failed a validation check.'))
    st.caption('Check claims against the passages. An "ok" status confirms formal checks, not factual accuracy.')
    if result.get('supporting_quote'):
        st.text('Supporting quote:')
        st.write(result['supporting_quote'])
    st.subheader('Retrieved sources')
    for source in result.get('context', []):
        with st.expander(f"[{source['passage_id']}] {source['date']} · {source['title']}"):
            st.text(source['text'])
            st.caption(f"Article ID: {source['record_id']} · Passage: {source['chunk_id']}")
