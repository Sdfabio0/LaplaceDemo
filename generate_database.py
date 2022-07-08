def generate_df(
    classes_file = 'employee_classes.pkl',
    n_lines = 200):

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

