import ast
import pickle
import random
import names
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
import fitz

true = True
list_dicts_str = []
list_dicts = []
list_dic_edges = []
shapes = {}
rule_edges = {}

def filter_constraints(list_sents):
    final = []
    for sent in list_sents:
        sent_2 = nltk.word_tokenize(sent)
        if 'should' in sent_2 or 'must' in sent_2:
            final.append(sent)
    return final


def extract_from_path(path):
    constraints = []
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
          text += page.get_text()

    sentences = nltk.sent_tokenize(text)
    sentences = [sent.replace('\n',' ') for sent in sentences]
    sentences = [sent.replace('\t',' ') for sent in sentences]
    sentences_fin = []

    grammars = [
    r"\d+[\.:]?\d*",
    r"[a-z]\)",
    r"i*\.",
    r"\.{2,}",
    r"\w\d",
    r"-",
    r"'",
    r" {2,}"]

    for idx,sent in enumerate(sentences):
        for idx2,grammar in enumerate(grammars):
            sent = re.sub(grammar," ",sent)
            if idx2 == len(grammars)-1 and len(nltk.word_tokenize(sent)) >= 5:
                sentences_fin.append(sent.strip())

    sentences_fin_1 = ['']

    for sent in sentences_fin:
        if sent[0] != sent[0].upper():
            sentences_fin_1[-1]+=' '+sent
        else:
            sentences_fin_1.append(sent)

    sentences_fin_2 = []
    lmtzr = WordNetLemmatizer()

    for sent in sentences_fin_1:
        s = nltk.word_tokenize(sent)
        s = [lmtzr.lemmatize(word) for word in s]
        # s = [lmtzr.lemmatize(word) for word in s if not word.lower() in stopwords.words()]
        final_sent = ' '.join(s)
        sentences_fin_2.append(final_sent.strip())

    constraints = filter_constraints(sentences_fin_2)
    return constraints,sentences_fin_2









def function_predicate_parse(dep,dictio):
    dic_edges = {}
    list_dics_used = []
    words = [dic['word'] for dic in dictio['words']]
    idxs_semi = [(m.start(0), m.end(0)) for m in re.finditer(':', dep)]
    idxs_words = re.findall('\d+',dep)
    idx_bra = dep.find('(')
    idx1 = idxs_semi[0]
    idx2 = idxs_semi[1]
    idxnum1 = re.findall('\d+',dep[idx_bra:idx1[0]])
    idxnum2 = re.findall('\d+',dep[idx1[0]:idx2[0]])
    if len(idxs_words) >=2:
        name = words[int(idxnum1[0])]
        target = words[int(idxnum2[0])]
        list_dics_used.append(str(dictio['words'][int(idxnum1[0])]))
        list_dics_used.append(str(dictio['words'][int(idxnum2[0])]))
        idx_bra = dep.find('(') #index of opening bracket '('
        if dep[:idx_bra] =='COUNT':
            name = 'COUNT'
        direction = 'name-target'
        edge = (name,target)
        edge_label = ''
        if '.' in dep[:idx_bra] or dep[idxs_semi[0][0]+1] =='e' or dep[idxs_semi[1][0]+1] =='e':
            funct = dep[:idx_bra]
            edge_label = funct
            target_dic = dictio['words'][int(idxnum2[0])]
            head_target_idx = target_dic['head']-1
            head_target_dic = dictio['words'][head_target_idx]
            name_dic = dictio['words'][int(idxnum1[0])]
            head_name_idx = name_dic['head']-1
            head_name_dic = dictio['words'][head_name_idx]
            if 'arg0' in funct or 'arg1' in funct:
                a=2
                direction = 'target-name'
                edge = (target,name)
            elif head_target_dic['word'] == name:
                if 'pass' in head_target_dic['dep'] or 'pass' in target_dic['dep']:
                    direction = 'target-name'
                    edge = (target,name)
            elif head_name_dic['word'] == target:
                if 'pass' in head_name_dic['dep'] or 'pass' in name_dic['dep']:
                    direction = 'target-name'
                    edge = (target,name)
            if dep[idxs_semi[1][0]+1] =='e' and dep[idxs_semi[0][0]+1] !='e' :
                direction = 'target-name'
                edge = (target,name)
            if name !=target :
                dic_edges = {'edge_name':edge,'edge_label':funct}
        if dep[int(idx1[0]+1)] == 's':
            func = 's'
            color_name = 'yellow'
            shape_name = 's'
        elif dep[int(idx1[0]+1)] == 'e':
            func = 'e'
            color_name = 'red'
            name = 'e.'+ name
            shape_name = 'o'
        if dep[int(idx2[0]+1)] == 'x':
            target =  target
            color_target = 'green'
            shape_target = 'o'
        if dep[int(idx1[0]+1)] == 'x':
            func = 's'
            color_name = 'Blue'
            shape_name = 's'
        elif dep[int(idx2[0]+1)] == 'e':
            func = 'e'
            color_target = 'red'
            target = 'e.'+ target
            shape_target = 'o'
        elif dep[int(idx2[0]+1)] == 's':
            func = 's'
            color_target = 'yellow'
            shape_target = 's'
        elif dep[int(idx2[0]+1)] == 'm':
            idx11 = dep.find('m.')+2
            idx12 = dep.find(')')
            target = dep[idx11:idx12]
            color_target = 'green'
            shape_target = 'o'
        dic = {
            'func':func,
            'name':name,
            'target':target,
            'color_name':color_name,
            'color_target':color_target,
            'shape_name':shape_name,
            'shape_target':shape_target,
            'direction': direction,
            'edge_label': edge_label
            }
    elif len(idxs_words)==1:
        func = 's'
        color_name = 'blue'
        color_target = 'blue'
        shape_name = shape_target = 'o'
        idx_bra = dep.find('(') #index of opening bracket '('
        name = dep[:idx_bra]
        target_idx = int(idxnum2[0])
        target = words[target_idx]
        list_dics_used.append(str(dictio['words'][int(idxnum2[0])]))
        direction = 'name-target'
        edge_label = ''
        dic = {
        'func':func,
        'name':name,
        'target':target,
        'color_name':color_name,
        'color_target':color_target,
        'shape_name':shape_name,
        'shape_target':shape_target,
        'direction': direction,
        'edge_label': edge_label
        }
    return dic,list_dics_used,dic_edges

def add_node_graph(G,dic):
    colors = {}
    name = dic['name']
    target = dic['target']
    func = dic['func']
    edge_label = ''
    if name not in list(G.nodes):
        G.add_nodes_from([(name, {"color": dic['color_name']})])
        colors[name] = dic['color_name']
        shapes[name] = dic['shape_name']
    if target not in list(G.nodes):
        G.add_nodes_from([(target, {"color": dic['color_target']})])
        colors[target] = dic['color_target']
        shapes[target] = dic['shape_target']
    #edge_label = dic['edge_label']
    edge_label = ''
    if dic['direction'] == 'name-target':
        G.add_edges_from([(name,target)],edge_labels=edge_label)
    elif dic['direction'] == 'target-name':
        G.add_edges_from([(target,name)],edge_labels=edge_label)
    return G,colors

def create_list_nodes(G):
    list_nodes = {}
    temp = list(G.nodes())
    while len(temp) > 0:
        el = temp[0]
        if el[0:2] == 'e.':
            search = el[2:]
            if search in temp:
                list_nodes[search] = ['event','entity']
                temp.remove(el)
                temp.remove(search)
            else:
                list_nodes[search] = ['event']
                temp.remove(el)
        else:
            search = 'e.' + el
            if search in temp:
                list_nodes[el] = ['event','entity']
                temp.remove(el)
                temp.remove(search)
            else:
                list_nodes[el] = ['entity']
                temp.remove(el)
    return list_nodes

def logical_graph(dic_work,G,redirect):
    colors = {}
    for dep in dic_work['dependency_lambda'][-1]:
        # for el in dep:
        dic_props, dics, dic_edges = function_predicate_parse(dep,dic_work)
        list_dic_edges.append(dic_edges)
        list_dicts_str.extend(dics)
        G,color_list = add_node_graph(G,dic_props)
        for key,color in color_list.items():
            colors[key] = color
    if redirect:
        list_nodes = create_list_nodes(G)

        for dic in set(list_dicts_str):
            child_dic = ast.literal_eval(dic)
            idx_head = child_dic['head']-1
            head_dic = dic_work['words'][idx_head]
            head = head_dic['word']
            child = child_dic['word']
            nodes = list(list_nodes.keys())
            event_head = 'e.' + head
            if head in nodes or event_head in nodes:
                types = list_nodes[head]
                heads = []
                childs = []
                for el in types:
                    if el == 'entity':
                        heads.append(head)
                    if el == 'event':
                        temp = 'e.' + head
                        heads.append(temp)
                for el2 in list_nodes[child]:
                    if el2 == 'entity':
                        childs.append(child)
                    if el2 == 'event':
                        temp = 'e.' + child
                        childs.append(temp)
                for el in heads:
                    for el2 in childs:
                        if G.has_edge(el,el2):
                            G.remove_edge(el,el2)
                            G.add_edge(el2,el)
                        elif  G.has_edge(el2,el):
                            continue
                        else:
                            continue
                           # G.add_edges_from([(el2,el)],edge_labels='',font_color='red')
    return G,colors,list_dic_edges

def build_rules(entities,G,dic_work):
    all_paths = []
    for ent1 in entities:
        for ent2 in entities:
            rule = []
            if ent1 != ent2:
                rule = func_path_rule(ent1,ent2,G,dic_work)
            if rule :
                all_paths.append(rule)
    return all_paths

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def function_name(name1,name2,dic_work,G):
    label = 'direct'
    name1 = name1.replace('e.','')
    if '.' in name1:
        idxs_points = [i for i, ltr in enumerate(name1) if ltr == '.']
        name1 = name1[idxs_points[-1]+1:]
    if (name1,name2) in list(G.edges):
        label = 'direct'
    func_name = name1.replace('e.','')
    idx2 = -1
    for idx,word_dic in enumerate(dic_work['words']):
        if name1 == word_dic['word']:
            idx1 = idx
        if name2 == word_dic['word']:
            idx2 = idx
    deps =dic_work['dependency_lambda'][-1]
    if idx2<0:
        for dep in deps:
            if str(idx1) in dep and name2 in dep:
                idx_bra = dep.find('(')
                func_name = dep[:idx_bra]
                idxs_points = [i for i, ltr in enumerate(func_name) if ltr == '.']
                if len(idxs_points)==1:
                    func_name = func_name[:idxs_points[0]]
                    func_name = func_name[:idxs_points[0]]
                    if func_name != name1:
                        func_name = name1 + dep[idxs_points[0]:idx_bra]
                elif len(idxs_points)==2:
                    idx_point1 = idxs_points[0]
                    idx_point2 = idxs_points[1]
                    func1 = func_name[:idx_point1]
                    func2 = func_name[idx_point2:]
                    func_name = func1 + func2.upper()
        return func_name
    if idx1 == idx2:
        return func_name
    for dep in deps:
        idxs_semi = [(m.start(0), m.end(0)) for m in re.finditer(':', dep)]
        idx1 = idxs_semi[0]
        idx2 = idxs_semi[1]
        idxnum1 = re.findall('\d+',dep[:idx1[0]])
        idxnum2 = re.findall('\d+',dep[idx1[0]:idx2[0]])
        if idx1 == int(idxnum1[0]) and idx2 == int(idxnum2[0]) :
            idx_bra = dep.find('(')
            func_name = dep[:idx_bra]
            idxs_points = [i for i, ltr in enumerate(func_name) if ltr == '.']
            if len(idxs_points)==1:
                func_name = func_name[:idxs_points[0]]
                if func_name != name1:
                    func_name = name1 + dep[idxs_points[0]:idx_bra]
            elif len(idxs_points)==2:
                idx_point1 = idxs_points[0]
                idx_point2 = idxs_points[1]
                func1 = func_name[:idx_point1]
                func2 = func_name[idx_point2:]
                func_name = func1 + func2.upper()
            if label =='direct' and name2 in func_name:
                func_name = name1.replace('e.','')
                return func_name
    return func_name


def func_path_rule(start,finish,G,dic_work):
    try:
        path = nx.shortest_path(G, source=start, target=finish)
    except nx.NetworkXNoPath:
        return []
    if len(path)>=3:
        rule = path[0]
        for idx,el in enumerate(path):
            if 'e.' in el:
                func_name = function_name(el,path[idx+1],dic_work,G)
                wrd = el.replace('e.','')
                test = 2
                for idx2,word in enumerate(dic_work['words']):
                    if wrd == word['word']:
                        if 'neg' == dic_work['words'][idx2-1]['dep'] or 'neg' == dic_work['words'][idx2-2]['dep'] :
                            func_name = 'NOT.' + func_name
                if func_name != path[idx-1]:
                    rule+='.'+func_name
            elif idx == 0:
                rule = el
            else:
                rule+=' --> '+ el
    else:
        rule = []
    return rule

def graph_dic(dic_work):
    G = nx.DiGraph()
    G,colors,_ = logical_graph(dic_work,G,redirect = False)
    entities = []
    #Generate the rules
    list_nodes = create_list_nodes(G)
    for key,arr in list_nodes.items():
        if 'entity' in arr:
            entities.append(key)

    rules = build_rules(entities,G,dic_work)

    #Plot the graph
    pos = nx.shell_layout(G,scale=3)
    font = 20/((len(list(G.nodes)))**(1/4))
    size_node = 2200/((len(list(G.nodes)))**(1/4))
    fig = plt.figure()
    plt.title(dic_work['sentence'],wrap=True)
    nx.draw(G, pos=pos, node_color = list(colors.values()),node_size = size_node,
    with_labels = True, font_size=font,node_shape ='o',arrows=True,edgecolors='black',width=2)  #   networkx draw()
    nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'edge_labels'), font_size=font)
    plt.savefig("/home/ff/Documents/UDepLambdaaIrudi/Graph.png", format="PNG")
    return rules,fig



def generate_df(
    classes_file = 'employee_classes.pkl',
    n_lines = 100):

    with open(classes_file,'rb') as f:
        class_list = pickle.load(f)

    gangs = []
    shifts_list = ['MORNING','AFTERNOON','NIGHT']
    salary_list = [30,40,20,35]
    seniority_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    available_shift = []
    data_keys = list(class_list.keys())
    p_classes = []
    s_classes = []
    name_list = []
    salaries = []
    seniorities = []
    for i in range(n_lines):
        idx1 = random.randint(0,len(data_keys)-1)
        primary_class = data_keys[idx1]
        idx2 = random.randint(0,len(class_list[primary_class])-1)
        secondary_class = class_list[primary_class][idx2][1]
        shift = random.choice(shifts_list)
        salary = random.choice(salary_list)
        seniority = random.choice(seniority_list)
        seniorities.append(seniority)
        salaries.append(salary)
        gang = random.randint(0,15)
        gangs.append(gang)
        p_classes.append(primary_class)
        s_classes.append(secondary_class)
        available_shift.append(shift)
        name_list.append(names.get_full_name())

    dic ={
        'Names':name_list,
        'Primary_Class': p_classes,
        'Secondary_Class': s_classes,
        'Shift_Availability': available_shift,
        'Gang': gangs,
        'Salaries': salaries,
        'Seniorities': seniorities
    }
    df = pd.DataFrame(data = dic)
    return df

