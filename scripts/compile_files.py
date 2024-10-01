import pandas as pd
import sys
import os

li = []
models=["GPS","Graph_transformer","GAT","GCN","SAGE"]
# models = ["GCN", "GAT","SAGE"]
print()
for num_classes in [7,5,3,2]:
    for degree_corrected in [True]:
        for type_of_model in models:
            df = pd.DataFrame()
            for i in range(200):
                file_name = f"runs/{num_classes}_{type_of_model}({i}).txt" if not degree_corrected else f"runs/{num_classes}_DC_{type_of_model}({i}).txt"
                new_data = pd.read_csv(file_name,sep=" ",header=None,names=["accs", "lamb", "mu"])
                if type_of_model == "Graph_transformer":
                    if len(new_data)!=121:
                        li.append(i)
                df = pd.concat([df,new_data],ignore_index=True)
                # print(df)
            file_name = f"compiled_maxes/{num_classes}_{type_of_model}.txt" if not degree_corrected else f"compiled_maxes/{num_classes}_DC_{type_of_model}.txt"
            df.to_csv(file_name,sep=" ",header=None,index=False)
            print(f"Wrote to {type_of_model} with a class size of {num_classes} and degree corrected: {degree_corrected}")
            print(li)
            
