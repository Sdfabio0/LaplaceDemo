from fastapi import FastAPI,UploadFile
import os
import json
from zipfile import ZipFile
from fastapi.responses import StreamingResponse, FileResponse
import matplotlib.pyplot as plt
import utils
import spacy_merge_phrases
import constraint_P





path = '/home/ff/Documents/UDepLambdaaIrudi'
app = FastAPI()


@app.post("/extract_rules")
def pdf_rules_extract(pdf_file: UploadFile):
    file_location = path + '/' +pdf_file.filename
    with open(file_location, "wb+") as file_object:
        file_object.write(pdf_file.file.read())
    constraints,sentences_fin_2 = utils.extract_from_path(file_location)
    return constraints

@app.get("/sent_to_logic_form")
def sent_to_logic_form(sent=None):
    sent = spacy_merge_phrases.generate_noun_phrase_sent(sent)
    print(sent)
    with open('input-english2.txt','w') as f:
        f.write(r'{"sentence":"'+sent+r'"}')
        f.write('\n')
    sent_file = 'input-english2.txt'
    output = 'output.json'
    #preprocessing of the sentence into a JSON file
    command = 'cat '+ sent_file +' | sh run-english.sh > '+ output
    os.system(command)
    # importing the dictionary
    # Opening JSON file
    with open('output.json') as json_file:
        dic_work = json.load(json_file)   
    return dic_work

@app.get("/logic_graph")
def logic_graph_processing(sent):
    dic = sent_to_logic_form(sent)
    _,fig = utils.graph_dic(dic) 
    fname = 'Graph.png'
    file_path = os.path.join(path,fname)
    fig.savefig(fname)  
    return FileResponse(file_path,media_type="image/png",filename=fname)
    
@app.get("/rules")
def generate_simple_rules(sent):
    dic = sent_to_logic_form(sent)
    rules,_ = utils.graph_dic(dic)
    return rules

@app.get("/filter_CP")
def filter_CP(sent=None, df_table= None,test=1):
    if sent != None:
        rules_all = generate_simple_rules(sent)
    if test:
        rules_all = []
        with open('logic_rules_test.txt','r') as f:
            line = f.readline()
            rules_all.append(line)
            print(line)
        f.close()	
    rules_npy = rules_all
    if df_table == None:
    	df_table = utils.generate_df()
    filtered_table,dic_sols = constraint_P.filter_database(rules_npy,df_table)
    print(filtered_table)
    print(dic_sols)
    return filtered_table
