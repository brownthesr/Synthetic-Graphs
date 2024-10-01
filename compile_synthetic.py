import numpy as np 
import pandas as pd 

df_list = []
for model in ["GCN","GAT","SAGE","GPS","GraphTransformer"]:
    acc_mean, acc_std = None,None
    for typ in ["Epsilon", "Heirarchical", "Triadic"]:
        data = np.loadtxt(f"{typ}_sbm_{model}_edges.txt")
        mu = data[:,-1]
        
        # fixer = 1.224744871391589
        for i in range(-1,2):
            # Create DataFrame for original data
            og_df = pd.DataFrame(data[:,0], columns=["Acc"])
            og_df["Transform Type"] = typ
            if typ == "Epsilon":
                og_df["Transform Type"] = "Geometric"
            og_df["Model"] = model
            og_df["Structured"] = True  # Indicates original data
            og_df["Mu"] = mu  # Indicates original data
            
            # Create DataFrame for modified data
            mod_df = pd.DataFrame(data[:,1], columns=["Acc"])
            mod_df["Transform Type"] = typ
            if typ == "Epsilon":
                mod_df["Transform Type"] = "Geometric"
            mod_df["Model"] = model
            mod_df["Structured"] = False  # Indicates modified data
            mod_df["Mu"] = mu  # Indicates modified data
            
            # Concatenate both DataFrames
            df_list.extend([mod_df,og_df])
df = pd.concat(df_list)
df.to_csv("compiled_maxes/structured.csv")