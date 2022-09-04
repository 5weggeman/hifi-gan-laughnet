import argparse
import json
from env import AttrDict
import glob
import os
import re
import shutil
import random
import sox
import multiprocessing

#Setting a seed for the random module
random.seed(1234)

#Initialising global variables
exclude=["p228","p229","p240","p257","p281","p299","p315","p329","p376","s5"]
path=os.getcwd()

train_set=[]
val_set=[]
val_split=0.15

#Allowing arguments to be parsed through the jobscript
parser = argparse.ArgumentParser()
parser.add_argument('--data', default="VCTK")
parser.add_argument('--config', default="config_v1.json")

a = parser.parse_args()

with open(a.config) as f:
  data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

#Ensuring one of two possible input options for data
assert a.data == "VCTK" or a.data == "laughter", "The only valid arguments are: 'VCTK' or 'laughter'."

def transform(split_ld):
  for item in split_ld:
    #The line below can be commented out to check the bitdepth of the recordings
    #print(f"bd: {sox.file_info.bitdepth(item)}")
    
    #Extracting the filename and creating a new file name and storage path for the new file
    fn = re.split(r"[/.]", item)[-2]
    nfn = fn+"_22k.wav"
    path_nf = os.path.join(path, "VCTK-0.92", "wavs", nfn)   
    
    #Creating a sox transformer object
    tfm = sox.Transformer()
    tfm.set_globals(guard = True)
    tfm.rate(h.sampling_rate,'v')
  
    #Saving the transformed file
    tfm.build_file(input_filepath = item, output_filepath = path_nf)

if a.data == "VCTK":
  #Preprocessing VCTK
  path_core="wav48_silence_trimmed"
  for folder_path in glob.glob(os.path.join(path,path_core,"**")):
    if os.path.isdir(folder_path):
      folder = re.split(r"/", folder_path)[-1]
      #Removing directories with data from 10 of the 110 speakers
      if folder in exclude:
        shutil.rmtree(folder_path)
      else:
        #Removing the mic2.flac files from the remaining directories
        for item2rm in glob.glob(os.path.join(folder_path, "*2.flac")):
          os.remove(item2rm)
        #Removing 40% of the mic1.flac files from the remaining directories to get to ~24.4 hours, excluding files from the rainbow passage (needed for validation)
        ld_folder = os.listdir(folder_path)
        rainbow_indices=[item1k for item1k in ld_folder if re.match(r".*0(0[1-9]|1[0-9]|2[0-4])\_mic1.flac$", item1k)]
        rest = list(set(ld_folder)-set(rainbow_indices))
        #Shuffling the remaining files in the folder to ensure that not the removed files are not consecutive
        random.shuffle(rest)
        #Splitting the rest files into files to remove and files to keep
        rest_rm = rest[0:round(0.4*len(ld_folder))]
        rest_keep = rest[round(0.4*len(ld_folder)):]
        #Actually removing the files
        for item1rm in rest_rm:
          os.remove(os.path.join(folder_path, item1rm))
        
        #Checking whether there are enough files for validation based on the val_split variable and adding files from the rest to fill the remainder of the validation set if needed
        len_rem = len(rainbow_indices)+len(rest_keep)
        curr_val_split = len(rainbow_indices)/len_rem
        rem_val=[]
        if not curr_val_split >= val_split:
          rem_val_split = val_split-curr_val_split
          rem_val = rest_keep[0:round(rem_val_split*(len_rem))]
          
        #Adding the subset of files to the validation set
        val_subset = rainbow_indices+rem_val
        val_subset = [re.split(r"[/.]", val_item)[-2] for val_item in val_subset]
        val_set = val_set+val_subset
        
        #Adding the subset of files to the training set
        train_subset = list(set(rest_keep)-set(rem_val))  
        train_subset = [re.split(r"[/.]", train_item)[-2] for train_item in train_subset]
        train_set = train_set+train_subset
  
  #Writing the names of the files (after transformation)      
  with open(os.path.join(path,"VCTK-0.92","validation.txt"), "w+") as file1:
    file1.writelines([fn+"_22k|\n" for fn in val_set])
  with open(os.path.join(path,"VCTK-0.92","training.txt"), "w+") as file2:
    file2.writelines([fn+"_22k|\n" for fn in train_set])

  #Creating a list of all flac files in the directory and splitting it to run this script as multiple tasks to speed up the process
  n_tasks = 24
  ld = glob.glob(os.path.join(path,path_core,"**","*.flac"), recursive=True)
  split_ld = [ld[x:x+int(len(ld)/n_tasks)] for x in range(0, len(ld), int(len(ld)/n_tasks))] 
  
  #Running the transform as multiple tasks to speed up the process
  poolSize = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
  pool = multiprocessing.Pool(processes=poolSize)
  pool.map(transform, split_ld)
  pool.close()
  pool.join()    