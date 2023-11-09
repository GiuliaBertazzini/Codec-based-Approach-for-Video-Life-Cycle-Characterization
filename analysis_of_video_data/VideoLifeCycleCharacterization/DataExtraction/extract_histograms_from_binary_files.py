import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--video_data_output_path', type=str, default='/Prove/Bertazzini/SOCIAL_DATASET', help='Location of output directory for H.264 generated videos and binary output files')
parser.add_argument('--file_zip_path', type=str)
parser.add_argument('--social', type=str)
parser.add_argument('--rewrite_bin_release_path', type=str, default='/data/lesc/staff/bertazzini/rewrite_h264_bin/target/release')
parser.add_argument('--save_path', type=str, default='/Prove/Bertazzini/SOCIAL_DATASET', help='Location to save txt file with binary files paths')
args = parser.parse_args()

def extractHistograms(video_data_output_path, rewrite_bin_release_path):
    print("Extracting Histograms...")
    for social in os.listdir(video_data_output_path):
        social_path = f"{video_data_output_path}/{social}"
        if os.path.isdir(social_path) and not isDirectoryEmpty(social_path):
            binary_outputs = os.listdir(social_path+"/BINARY_OUTPUTS")
            for binary_file in binary_outputs:
                print(f"VIDEO: {binary_file.replace('_binary_output.zip', '_')} - SOCIAL: {social}")
                rewriteBinFile(f"{social_path}/BINARY_OUTPUTS", binary_file, rewrite_bin_release_path, social=social)

def rewriteBinFile(files_zip_path, filename, rewrite_bin_release_path, social):
    binary_outputs_dir = files_zip_path
    video_name = filename.replace("_binary_output.zip", "")

    file_zip_path = f"{binary_outputs_dir}/{filename}"

    with zipfile.ZipFile(file_zip_path, 'r') as zip_file:
        zip_file.extractall(binary_outputs_dir)

    original_bin_path = f"{binary_outputs_dir}/binary_output.xml"
    json_file_name = f"{video_name}_histogram.json"
    graph_file_name = f"{video_name}_graph.json"
    os.chdir(rewrite_bin_release_path)
    cmd = f'./rewrite_bin {original_bin_path} {json_file_name} {graph_file_name}'
    os.system(cmd)

    print('Histogram generated')
    os.makedirs(f"/Prove/Bertazzini/SOCIAL_DATASET/{social}/HISTOGRAMS", exist_ok=True)
    cmd = f'mv {json_file_name} /Prove/Bertazzini/SOCIAL_DATASET/{social}/HISTOGRAMS'
    os.system(cmd)
    print('File moved')

    os.chdir(binary_outputs_dir)
    deleteFiles(binary_outputs_dir)

def compressFile(file_path, video_name):
    optimized_binary_file_dir = file_path.replace("BINARY_OUTPUTS", "BINARY_OUTPUTS_OPTIMIZED")
    os.makedirs(optimized_binary_file_dir, exist_ok=True)

    binary_filename = "binary_output_optimized.xml"
    binary_file_extension = binary_filename[binary_filename.find(".")+1:]
    zip_name = video_name + "_" + binary_filename.replace(binary_file_extension, "zip")
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(file_path+f"/{binary_filename}", arcname=os.path.basename(file_path))
    return zip_name

def deleteFiles(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        if file in ['binary_output.xml']: #, 'binary_output_optimized.xml']:
            os.remove(os.path.join(dir_path, file))

def isDirectoryEmpty(directory_path):
    if len(os.listdir(directory_path)) == 0:
        return True
    else:
        return False

def createTxtWithFilesPath(video_data_output_path, save_path):
    for social in os.listdir(video_data_output_path):
        binary_outputs_dir = video_data_output_path + f"/{social}/BINARY_OUTPUTS"
        for file in os.listdir(binary_outputs_dir):
            bin_path = binary_outputs_dir + f"/{file}"
            with open(f'{save_path}/binary_outputs_path.txt', 'a') as txt_file:
                txt_file.write(f'{bin_path} {social} \n')

def generateHistograms(file_zip_path, social, rewrite_bin_release_path):
    video_name = file_zip_path.split("/")[-1].replace("_binary_output.zip", "")

    with tempfile.TemporaryDirectory(dir='/Prove/Bertazzini/Temp') as temp_path:
        shutil.copy(file_zip_path, temp_path)

        with zipfile.ZipFile(temp_path+f"/{video_name}_binary_output.zip", 'r') as zip_file:
            zip_file.extractall(temp_path)

        original_bin_path = f"{temp_path}/binary_output.xml"
        json_file_name = f"{video_name}_histogram.json"
        graph_file_name = f"{video_name}_graph.json"

        cmd = ['./rewrite_bin', f'{original_bin_path}', f"{json_file_name}", f"{graph_file_name}"]
        subprocess.run(cmd, cwd=rewrite_bin_release_path, shell=False)

        print('Histogram Generated!')
        os.makedirs(f"/Prove/Bertazzini/SOCIAL_DATASET/{social}/HISTOGRAMS", exist_ok=True)
        shutil.move(rewrite_bin_release_path+f"/{json_file_name}", f"/Prove/Bertazzini/SOCIAL_DATASET/{social}/HISTOGRAMS")
        print('File json moved!')

if __name__ == '__main__':
    #createTxtWithFilesPath(args.video_data_output_path, args.save_path)
    generateHistograms(args.file_zip_path, args.social, args.rewrite_bin_release_path)
