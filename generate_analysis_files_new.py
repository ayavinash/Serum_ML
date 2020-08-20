### This script filters the maxquant proteingroups file to discard genes
### which are not quantified in specified proportion of replicates 
### in each condition



import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os
import sys

def get_proteins(expdesign,protein_groups,filter_val,intensity):
    sep="\t"
    expdesign = pd.read_csv(expdesign,sep=sep)
    data = pd.read_csv(protein_groups,sep=sep)
    data.columns = data.columns.str.replace(" ","_")
    data = data[~(data.Gene_names.isnull() & data.Protein_IDs.isnull())]
    data= data[data.Reverse.isnull()]
    data = data[data.Potential_contaminant.isnull()]
    print(expdesign.columns)
    if intensity=="LFQ":
        print("Using LFQ Intensity....")
        expdesign["lfq_cols"]= "LFQ_intensity_"+expdesign.label
    else:
        print("Using RAW Intensity...")
        expdesign["lfq_cols"]= "Intensity_"+expdesign.label
    lfq_data = data[expdesign.lfq_cols]
    lfq_data = lfq_data.replace(0,np.nan)
    print(expdesign.groupby("condition")["label"].count()*filter_val)
    valid_counts =(expdesign.groupby("condition")["label"].count()*filter_val).apply(lambda x: int(x)).to_dict()
    print("Valid samples...", valid_counts)
    for condition in valid_counts:
        lfq_data["in_"+condition] = lfq_data[expdesign.lfq_cols[expdesign.condition==condition]].apply(
                                    lambda x: x[x.notnull()].count(),axis=1)
        lfq_data["in_"+condition+"_ok"] = lfq_data[expdesign.lfq_cols[expdesign.condition==condition]].apply(
                                    lambda x: x[x.notnull()].count()>=valid_counts[condition],axis=1)
    ok_cols = ["in_"+condition+"_ok" for condition in valid_counts]
    lfq_data["is_ok"]= lfq_data[ok_cols].apply(lambda x: True if x[x].count()>0 else False,axis=1)
    if intensity=="LFQ":
        lfq_cols = expdesign.lfq_cols.str.replace("LFQ_intensity_","LFQ intensity ")
    else:
        lfq_cols = expdesign.lfq_cols.str.replace("Intensity_","Intensity ")

    return lfq_data[lfq_data.is_ok].index,lfq_cols

def write_filtered_file(protein_groups,valid_index,filter_val,out_dir,expdesign,intensity,lfq_cols):
    read_sep="\t"
    write_sep="\t"
    if write_sep==",":
        ext=".csv"
    else:
        ext=".txt"
    data = pd.read_csv(protein_groups,sep=read_sep)
    expdesign_name = os.path.splitext(os.path.basename(expdesign))[0]
    proteinGroups_name = os.path.splitext(os.path.basename(protein_groups))[0]
    outfile_path = os.path.join(out_dir,proteinGroups_name+"_"+intensity+"_"+expdesign_name+"_"+str(filter_val)+ext)
#    outfile_path = os.path.join(out_dir,"proteinGroups_"+expdesign_name+"_"+str(filter_val)+".txt")
    required_cols = ["Gene names","Protein IDs","Reverse","Potential contaminant"] +lfq_cols.to_list()
    

    data.loc[valid_index,required_cols].to_csv(outfile_path,sep=write_sep,index=False)

    print("Writing filtered proteinGroups file to...",outfile_path)
    print("Done...")



def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-p",dest="protein_groups",
            help ="full path to protein groups file...",type=str,
            default=os.path.join(os.getcwd(),"proteinGroups.txt"))
    parser.add_argument("-e",dest="expdesign",
            help="full path to expdesign file...",type=str,default=None)
    parser.add_argument("-o",dest="out",
            help="full path to output folder...",type=str,default= os.getcwd())
    parser.add_argument("-f",dest ="filter",help="filter fraction",default =0.7,type=float)
    parser.add_argument("-dir",dest ="dir",help="directory with exp design files",default =None)
    parser.add_argument("-i",dest ="intensity",help="Raw or LFQ",default ="LFQ",type=str,choices=["LFQ","RAW"])

    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if not args.dir and not args.expdesign:
        sys.exit("Provide an exp design file or a folder with exp design files...!")
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if args.dir and os.path.isdir(args.dir):
        expdesign_files = [x for x in os.listdir(args.dir) if os.path.splitext(x)[1]==".txt"]
        for expdesign_file in expdesign_files:
            print("processing file...",expdesign_file)
            expdesign_path = os.path.join(args.dir,expdesign_file)
            valid_index,lfq_cols = get_proteins(expdesign_path,args.protein_groups,args.filter,args.intensity)
            write_filtered_file(args.protein_groups,valid_index,args.filter,args.out,expdesign_path,args.intensity,lfq_cols)
    
    else:
        valid_index,lfq_cols = get_proteins(args.expdesign,args.protein_groups,args.filter,args.intensity)
        write_filtered_file(args.protein_groups,valid_index,args.filter,args.out,args.expdesign,args.intensity,lfq_cols)

if __name__=="__main__":
    main()

