import os
import json
import matplotlib.pyplot as plt
import utils
import spacy_merge_phrases
import constraint_P
import streamlit as st
import base64
import plotly.graph_objects as go

path = '/home/ff/Documents/UDepLambdaaIrudi'

def pdf_rules_extract(pdf_file):
    file_location = path + '/' +pdf_file
    constraints,sentences_fin_2 = utils.extract_from_path(file_location)
    return constraints

def sent_to_logic_form(sents):
    in_sent = []
    for sent in sents:
    	sent = spacy_merge_phrases.generate_noun_phrase_sent(sent)
    	in_sent.append(sent)
    with open('input-english2.txt','w') as f:
    	for sen in in_sent:
            f.write(r'{"sentence":"'+sen+r'"}')
            f.write('\n')
    sent_file = 'input-english2.txt'
    output = 'output.json'
    #preprocessing of the sentence into a JSON file
    command = 'cat '+ sent_file +' | sh run-english.sh > '+ output
    os.system(command)
    # importing the dictionary
    # Opening JSON file
    with open('output.json') as json_file:
        dics_work = [json.loads(line) for line in json_file]  
    return dics_work
    
def generate_simple_rules(sents):
    dics = sent_to_logic_form(sents)
    rules = []
    for dic in dics:
        rule_set,_ = utils.graph_dic(dic)
        rules.append(rule_set)
    print(rules) 
    return rules

def filter_CP(rules, df_table, sent=None, test=0):
    if sent != None:
        rules_all = generate_simple_rules(sent)
    if test:
        rules_all = []
        with open('logic_rules_test.txt','r') as f:
            for line in f:
            	rules_all.append(line)
        f.close()
    if rules:
    	rules_all = rules	
    rules_npy = rules_all
    filtered_table = constraint_P.filter_database(rules_npy,df_table)
    return filtered_table
    
def read_rules(num):
    out_sentences = []
    rules_tot = []
    file_rule = 'input-test-en-' + str(num) + '.txt'
    with open(file_rule,'r') as f:
    	for line in f:
        	out_sentences.append(line)
    f.close()
    print(out_sentences)
    rules_tot = generate_simple_rules(out_sentences)
    return rules_tot

st.set_page_config(layout='wide')  

st.title('Choose a Document to filter the table')



with st.container():
    col1, col2, col3 = st.columns(3)
    doc_choose = -1
    # this will put a button in the middle column
    with col1:
        with open('PDF-converted-input-test-en-1.pdf',"rb") as f:
        	base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="200" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        if st.button('Filter with Doc1'):
        	doc_choose = 1

    with col2:
        with open('PDF-converted-input-test-en-2.pdf',"rb") as f:
        	base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="200" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        if st.button('Filter with Doc2'):
        	doc_choose = 2

    with col3:
        with open('PDF-converted-input-test-en-1.pdf',"rb") as f:
        	base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="200" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        if st.button('Filter with Doc3'):
        	doc_choose = 2


pdf_name = 'articles.pdf'


table = utils.generate_df()

st.title('Primary table: ' + str(len(table)) + ' employees total')
st.write(table)
if doc_choose == 1 or doc_choose == 2 or doc_choose == 3:
	rules_tot = read_rules(doc_choose)
	print(rules_tot)
	final_table = filter_CP(rules = rules_tot,df_table = table)
	st.title('Filtered Table: ' + str(len(final_table)) + ' employees remaining')
	st.write(final_table)
