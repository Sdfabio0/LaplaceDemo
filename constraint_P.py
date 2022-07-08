import re
import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
import json
from ortools.sat.python import cp_model
import numpy as np
from nltk.stem import WordNetLemmatizer
import pickle
import random
import names

class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        filename = 'solutions.json'
        with open(filename, "r") as file:
            dic_sols = json.load(file)
        for v in self.__variables:
            #print('%s=%i' % (v, self.Value(v)), end=' ')
            if v._IntVar__negation == None:
                dic_sols[v._IntVar__var.name].append(self.Value(v))
        with open('solutions.json', 'w') as fp:
            json.dump(dic_sols, fp)

        if self.__solution_count >= self.__solution_limit:
            #print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count

lemmatizer = WordNetLemmatizer()

def flatten(d):
    res = []  # Result list
    if isinstance(d, dict):
        for key, val in d.items():
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        raise TypeError("Undefined type for flatten: %s"%type(d))
    return res




def filter_database(rules_npy,df):

    filename = 'solutions.json'
    dic_sols = {
            'Names':[],
            'Primary_Class': [],
            'Secondary_Class': [],
            'Shift_Availability': [],
            'Gang': [],
            'Salaries': [],
            'Seniorities': []
        }
    with open('solutions.json', 'w') as fp:
                json.dump(dic_sols, fp)
                
    rules = rules_npy
    df.to_excel('employees.xlsx', engine='xlsxwriter')
    #Encoding the data
    le = preprocessing.LabelEncoder()
    d = defaultdict(preprocessing.LabelEncoder)
    df_encode = df.apply(le.fit_transform)
    fit = df.apply(lambda x: d[x.name].fit_transform(x))
    #print(df_encode.head(3))
    ## Decoding
    decode = df_encode.apply(lambda x: d[x.name].inverse_transform(x))
    # print(decode.head(3))
    dic_cols = {}
    dic_vars= {}
    model = cp_model.CpModel()


    vars = [None]*len(df.columns)
    dic_cols_raw = {}
    for idx,col in enumerate(df.columns):
        dic_cols[col] = list(np.unique(df_encode[col]))
        dic_cols_raw[col] = list(np.unique(df[col]))
        vars[idx] = model.NewIntVar(0, max(dic_cols[col]), col)
        dic_vars[col] = vars[idx]
        # vars[idx].SetValues(dic_cols[col])

    lines = []
    keywords = ['less','more','equal','times']
    counts = {}
    # bools = [None]*len(rules)
    bools = {}
    const_dic_total = []
    for idx,rule_set in enumerate(rules):
        const_dic_all = []
        dic_bools = {}
        for col in list(df.columns):
            dic_bools[col] = []
        if len(rule_set):
            # bools[idx] = model.NewBoolVar(str(idx))
            for idx2,rule_unit in enumerate(rule_set):
                key_ = None
                nott = False
                if 'NOT' in rule_unit:
                    nott = True
                for key in keywords:
                    if key in rule_unit:
                        key_ = key
                nums = re.findall('\d+',rule_unit)
                const_dic = {}
                if nums:
                    nums = [int(num) for num in nums]
                    for col in list(df.columns):
                        if lemmatizer.lemmatize(col.lower()) in rule_unit:
                            const_dic[col] = int(nums[0])
                            break
                    if not bool(const_dic):
                        testt = 1
                        # counts = [None]*(max(nums)-min(nums))
                        # for i in range(max(nums)-min(nums)):
                        #     counts[i] =model.NewIntVar(1, 1,'bool'+str(idx))
                        # model.Add(sum(counts) >= min(nums)).OnlyEnforceIf(bools[idx])
                        # model.Add(sum(counts) <= max(nums)).OnlyEnforceIf(bools[idx])
                    else:
                        if key_ == 'equal' or key_ == None:
                            dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '==' + str(const_dic[col])))
                            # model.Add(dic_vars[col] == const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                            const_dic_all.append(str(dic_vars[col]) + '==' + str(const_dic[col]))
                        elif key_ == 'less':
                            dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '<=' + str(const_dic[col])))
                            # model.Add(dic_vars[col] <= const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                            const_dic_all.append(str(dic_vars[col]) + '<=' + str(const_dic[col]))
                        else:
                            dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '>=' + str(const_dic[col])))
                            # model.Add(dic_vars[col] >= const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                            const_dic_all.append(str(dic_vars[col]) + '>=' + str(const_dic[col]))
                else:
                    for col,vals in dic_cols_raw.items():
                        if col == 'Names' or col =='Gang' or col == 'Seniorities' or col=='Secondary_Class' :
                            continue
                        for val in vals:
                            val_new = val
                            if str(val_new) == val_new:
                                val_new = val_new.replace(' ', '')
                            if lemmatizer.lemmatize(str(val_new).lower()) in rule_unit.lower():
                                const =  pd.DataFrame(data=[val]).apply(lambda x: d[col].transform(x))
                                const_dic[col] = int(const.values)
                    for col,val in const_dic.items():
                        if nott:
                            if key_ == 'equal' or key_ == None:
                                dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '==' + str(const_dic[col])))
                                # model.Add(dic_vars[col] == const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                                const_dic_all.append(str(dic_vars[col]) + '!=' + str(const_dic[col]))
                            elif key_ == 'less':
                                dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '<=' + str(const_dic[col])))
                                # model.Add(dic_vars[col] <= const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                                const_dic_all.append(str(dic_vars[col]) + '>=' + str(const_dic[col]))
                            else:
                                dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '>=' + str(const_dic[col])))
                                # model.Add(dic_vars[col] >= const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                                const_dic_all.append(str(dic_vars[col]) + '<=' + str(const_dic[col]))
                        else:
                            if key_ == 'equal' or key_ == None:
                                dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '==' + str(const_dic[col])))
                                # model.Add(dic_vars[col] == const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                                const_dic_all.append(str(dic_vars[col]) + '==' + str(const_dic[col]))
                            elif key_ == 'less':
                                dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '<=' + str(const_dic[col])))
                                # model.Add(dic_vars[col] <= const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                                const_dic_all.append(str(dic_vars[col]) + '<=' + str(const_dic[col]))
                            else:
                                dic_bools[col].append(model.NewBoolVar(str(dic_vars[col]) + '>=' + str(const_dic[col])))
                                # model.Add(dic_vars[col] >= const_dic[col]).OnlyEnforceIf(dic_bools[col][-1])
                                const_dic_all.append(str(dic_vars[col]) + '>=' + str(const_dic[col]))
        bools[idx] = dic_bools
        const_dic_total.append(const_dic_all)

    const_dics = []
    for set_rules in const_dic_total:
        dic = {}
        for col in list(df.columns):
            dic[col] = []
        for constraint in set_rules:
            num = re.findall('\d+',constraint)
            idx_num = constraint.index(num[0])
            idx = idx_num - 2
            col = constraint[0:idx]
            if constraint[idx:] not in dic[col]:
                dic[col].append(constraint[idx:])
        dic_filter = dict( [(k,v) for k,v in dic.items() if len(v)>0])
        const_dics.append(dic_filter)

    const_dics_new = []
    print(const_dics)
    for dic in const_dics:
        dic_new = dic
        for key,arr in dic.items():
            if len(arr)>1:
                nums = []
                for constr in arr:
                    num = re.findall('\d+',constraint)
                    num = int(num[0])
                    if num not in nums:
                        nums.append(num)
                if len(nums) ==1:
                   for constr in arr:
                       if '<=' in constr:
                           dic_new[key] = [constr]
                       elif '>=' in constr:
                           dic_new[key] = [constr]
        const_dics_new.append(dic_new)
    print(const_dics_new)
    list_dic_bools = []
    dic_boools = {}
    for dic in const_dics_new:
        dic_bools = {}
        for col in list(df.columns):
            dic_bools[col] = []
        for key , arr in dic.items():
            for idx, constr in enumerate(arr):
                if '==' in constr:
                    num = re.findall('\d+',constr)
                    num = int(num[0])
                    name_bool = str(key) + '==' +str(num)
                    dic_bools[key].append(model.NewBoolVar(name_bool))
                    model.Add(dic_vars[key] == num).OnlyEnforceIf(dic_bools[key][-1])
                    model.Add(dic_vars[key] != num).OnlyEnforceIf(dic_bools[key][-1].Not())
                elif '<=' in constr:
                    num = re.findall('\d+',constr)
                    num = int(num[0])
                    name_bool = str(key) + '<=' +str(num)
                    dic_bools[key].append(model.NewBoolVar(name_bool))
                    model.Add(dic_vars[key] <= num).OnlyEnforceIf(dic_bools[key][-1])
                    model.Add(dic_vars[key] > num).OnlyEnforceIf(dic_bools[key][-1].Not())
                elif '>=' in constr:
                    num = re.findall('\d+',constr)
                    num = int(num[0])
                    name_bool = str(key) + '>=' +str(num)
                    dic_bools[key].append(model.NewBoolVar(name_bool))
                    model.Add(dic_vars[key] >= num).OnlyEnforceIf(dic_bools[key][-1])
                    model.Add(dic_vars[key] < num).OnlyEnforceIf(dic_bools[key][-1].Not())
                elif '!=' in constr:
                    num = re.findall('\d+',constr)
                    num = int(num[0])
                    name_bool = str(key) + '!=' + str(num)
                    dic_bools[key].append(model.NewBoolVar(name_bool))
                    model.Add(dic_vars[key] != num).OnlyEnforceIf(dic_bools[key][-1])
                    model.Add(dic_vars[key] == num).OnlyEnforceIf(dic_bools[key][-1].Not())
        dic_filter = dict([(k,v) for k,v in dic_bools.items() if len(v)>0])
        list_dic_bools.append(dic_filter)

    for dic in list_dic_bools:
        keys = list(dic.keys())
        if keys:
            prod = len(dic[keys[0]])*len(dic[keys[1]])
            if keys[0] == 'Shift_Availability':
                for bool1 in dic[keys[0]]:
                    for bool2 in dic[keys[1]]:
                        model.Add(bool1 == 1).OnlyEnforceIf(bool2)
            else:
                for bool1 in dic[keys[1]]:
                    for bool2 in dic[keys[0]]:
                        model.Add(bool1 == 1).OnlyEnforceIf(bool2)


    tot = []
    for dic in list_dic_bools:
        result = flatten(dic)
        tot = tot + result

    solver = cp_model.CpSolver()
    # Force the solver to follow the decision strategy exactly.
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Search and print out all solutions.
    allvars = vars + tot
    solution_printer = VarArraySolutionPrinterWithLimit(allvars,1)
    # solver.Solve(model, solution_printer)
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    model.ExportToFile('const.txt')
    for index, row in df_encode.iterrows():
        constraints_all = {}
        for col in df.columns:
            constraints_all[col] = model.Add(dic_vars[col] == row[col])
        status = solver.Solve(model, solution_printer)
        if solver.StatusName(status) == 'INFEASIBLE':
            # print('Status = %s' % solver.StatusName(status))
            # print(row)
            # print('\n')
            # print('\n')
            for col in df.columns:
                constraints_all[col].Proto().Clear()
        else:
            # print('FEASIBLE')
            # print('\n')
            # print('\n')
            # print(row)
            for col in df.columns:
                constraints_all[col].Proto().Clear()

    # print('Status = %s' % solver.StatusName(status))
    # print('Number of solutions found: %i' % solution_printer.solution_count())

    filename = 'solutions.json'

    with open(filename, "r") as file:
        dic_sols = json.load(file)

    data_sols_encode = pd.DataFrame(data = dic_sols)
    data_sols_decode = data_sols_encode.apply(lambda x: d[x.name].inverse_transform(x))
    data_sols_decode.to_excel('filtered_employees.xlsx', engine='xlsxwriter')
    return data_sols_decode
