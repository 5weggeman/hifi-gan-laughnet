#Imports
import argparse
import re
import sox
import os
import glob
import multiprocessing
import random

#Setting a seed for the random module
random.seed(1234)

#Initialising global variables
sr = 22050
complete_dir_list=[]
path_core="/scratch/s5007453/"

#Ensuring one of two possible input options
parser = argparse.ArgumentParser()
parser.add_argument('--data', default="VCTK")
a = parser.parse_args()

assert a.data == "VCTK" or a.data == "laughter", "The only valid arguments are: 'VCTK' or 'laughter'."

def preprocessing(sl):
#Looping over all items in the sublist
  for item in sl:
    #The line below can be commented out to check the bitdepth of the recordings
    #print(f"bd: {sox.file_info.bitdepth(item)}")
    
    #Extracting the filename and creating a new file name and storage path for the new file
    fn = re.split(r"[/.]", item)[-2]
    nfn = fn+"_22k.wav"
    path_nf = path_out+"/"+nfn   
    
    #Creating a sox transformer object
    tfm = sox.Transformer()
    tfm.set_globals(guard = True)
    tfm.rate(sr,'v')
  
    #Saving the transformed file
    tfm.build_file(input_filepath = item, output_filepath = path_nf)
    

if a.data == "VCTK":
#Preprocessing the VCTK dataset
  path=path_core+"wav48_silence_trimmed"
  path_out=path_core+"hifi-gan/VCTK-0.92/wavs"
  path_tr=path_core+"hifi-gan/VCTK-0.92/training.txt"
  path_val=path_core+"hifi-gan/VCTK-0.92/validation.txt"
  
  #Creating a directory list containing all the file paths
  dir_list=os.listdir(path)
  
  #Looping over all folders in the directory
  for folder in dir_list:
    if os.path.isdir(os.path.join(path, folder)):
      path2 = os.path.join(path, folder)
      dir_list2 = glob.glob(os.path.join(path2, "*1.flac"))
      complete_dir_list.extend(dir_list2)
      
  #Splitting the list to run this script as multiple tasks to speed up the process
  n_tasks = 24
  split_list = [complete_dir_list[x:x+int(len(complete_dir_list)/n_tasks)] for x in range(0, len(complete_dir_list), int(len(complete_dir_list)/n_tasks))]
  
  #Running the preprocessing as multiple tasks to speed up the process
  poolSize = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
  pool = multiprocessing.Pool(processes=poolSize)
  pool.map(preprocessing, split_list)
  pool.close()
  pool.join()
  
  #Shuffling the complete directory list for the following step
  random.shuffle(complete_dir_list)
  
  #Splitting the files in training and validation data by creating two text files with the respective file names
  with open(path_tr, "w+") as file1:
    file1.writelines([(re.split(r"[/.]", fp1)[-2]+"_22k|\n") for fp1 in complete_dir_list[0:round(0.99*len(complete_dir_list))]])
  with open(path_val, "w+") as file2:
    file2.writelines([(re.split(r"[/.]", fp2)[-2]+"_22k|\n") for fp2 in complete_dir_list[round(0.99*len(complete_dir_list)):]])
  
else:
#Preprocessing the extracted laughter samples from the MULAI database
  path=path_core+"laughter"
  path_out=path_core+"laughter/output"
  
  #Creating a directory list containing all the file paths
  complete_dir_list = glob.glob(os.path.join(path,"*.flac"))
  
  #Running the preprocessing
  preprocessing(complete_dir_list)