import pandas as pd
import ast
import numpy as np

def excel_to_python(file_name):
    def str_to_tuple_float(s):
        try:
            t = ast.literal_eval(s)
            return tuple(float(x) for x in t)
        except (ValueError, SyntaxError):
            return None  # or handle the error as needed

    df = pd.read_excel(file_name)
    df = df.dropna() # remove rows with missing values
    df = df.drop_duplicates() # remove duplicate rows
    df = df.map(str_to_tuple_float)
    df=df.to_numpy()
    df=df[:,1:]
    return(df)
def Calcul_onde(df):
    # Calcul de l'onde


    return None
#Test
S_parameters=excel_to_python("C:\\Users\\maell\\Desktop\\ENAC\\Projet SAT\\Filtered_TE_SMatrix.xlsx")
print("S_parameters: \n", S_parameters)