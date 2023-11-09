import os
import glob
import zipfile
import shutil
import argparse
import csv
import subprocess
import tempfile
from colorama import Fore

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file_path', type=str, default='/data/lesc/staff/bertazzini/VideoData/DATASET_CSV', help='Location of csv file containing videos paths')
parser.add_argument('--video_path', type=str, help="Path of a single video taken from the txt file")
parser.add_argument('--social', type=str, help="Social of the video")
parser.add_argument('--video_data_output_path', type=str, default='/Prove/Bertazzini/SOCIAL_DATASET', help='Location of output directory for H.264 generated videos and binary output files; default is the same of dataset subdirectories')
parser.add_argument('--jm19_bin_path', type=str, default='/data/lesc/staff/bertazzini/VideoData/h264-simple-analysis/bins', help='Location of the JM 19.0 bin directory')
parser.add_argument('--generate_h264', type=bool, default=False, help='if set to true it generates also the .h264 files, starting from the .mp4')
args = parser.parse_args()

FILE_TO_REMOVE = ["binary_output.xml", "binary_output_displayorder.xml", "dataDec.txt", "log.dec", "test_dec.yuv"]
SOCIAL_LIST = ['native', 'Youtube', 'Instagram', 'Twitter', 'Facebook']

def generateTXTFromCSV(csv_file_path):
    with open(f'{csv_file_path}/all_videos_avidemux.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            video_path = row[0]
            if video_path[0] != '/':
                start = video_path.find("/")
                video_path = video_path[start:]
            social = row[1]
            with open(f'{csv_file_path}/all_videos.txt', 'a') as txt_file:
                txt_file.write(f'{video_path} {social} \n')

def generateTXT(file_path, save_path):
    #generate a txt with the video path (in this case the .h264 files) and social
    for social in os.listdir(file_path):
        social_dir = file_path+f"/{social}"
        for video in os.listdir(social_dir):
            with open(f'{save_path}/videos_path_and_social.txt', 'a') as txt_file:
                txt_file.write(f'{social_dir}/{video} {social} \n')
def extractVideoData(video_path, social, video_data_output_path, jm19_bin_path, generate_h264):
    os.makedirs(f'{video_data_output_path}/{social}', exist_ok=True)
    if generate_h264:
        generateH264(video_path, social, video_data_output_path)
    decodeH264(video_data_output_path, video_path.split("/")[-1][:-5], social, jm19_bin_path)

def generateH264(video_path, social, video_data_output_path):
    # check if some videos in the directory are already in H.264 format
    h264_files = {}
    video_to_process = False

    existing_h264 = glob.glob(video_data_output_path + "/" + social  + "/*.h264")

    for video in existing_h264:
        h264_files[video.split("/")[-1].replace(".h264", "")] = True

    video_name = video_path.split("/")[-1][:-4]

    if video_name not in h264_files:
        video_to_process = True

    # PROCESS VIDEOS
    if video_to_process:
        input_video = video_path
        output_video = f"{video_data_output_path}/{social}/{video_name}.h264"
        cmd = "ffmpeg -hide_banner -loglevel error -i " + input_video + " -vcodec copy -an -bsf:v h264_mp4toannexb " + output_video
        os.system(cmd)

def decodeH264(video_data_output_path, video_name, social, jm19_bin_path):
    # check if some videos in the directory have already been decoded
    zip_files = {}

    os.makedirs(video_data_output_path + f"/{social}/BINARY_OUTPUTS", exist_ok=True)
    existing_zip = glob.glob(video_data_output_path + f"/{social}/BINARY_OUTPUTS" + "/*.zip")

    for zip in existing_zip:
        zip_files[zip.split("/")[-1].replace("_binary_output.zip", "")] = True

    video_to_decode = False

    if video_name not in zip_files:
        video_to_decode = True

    # DECODE VIDEOS
    if video_to_decode:
        with tempfile.TemporaryDirectory(dir='/Prove/Bertazzini/Temp') as temp_path:
            shutil.copy(jm19_bin_path + "/decoder.cfg", temp_path)

            # generating binary file
            cmd =[f'{jm19_bin_path}/ldecod.exe', '-i', f"{video_data_output_path}/{social}/{video_name}.h264"] #"./ldecod.exe -i " + video_data_output_path +"/" + social + "/" + video_name +".h264"
            print("Generating binary output for video " + video_name +"\n")
            subprocess.run(cmd, cwd=temp_path, shell=False)

            # compressing binary file
            print("COMPRESSING BINARY OUTPUT FILE..." )
            zip_path = compressFile(temp_path +"/" + "binary_output.xml", video_name=video_name)

            # moving binary file
            shutil.move(zip_path, video_data_output_path + f"/{social}/BINARY_OUTPUTS")

            print(Fore.LIGHTBLUE_EX+"BINARY FILE MOVED CORRECTLY!\n"+Fore.RESET)


def compressFile(file_path, video_name):
    temp_path, binary_filename = os.path.split(file_path)
    binary_file_extension = binary_filename[binary_filename.find(".")+1:]
    zip_path = temp_path+"/"+video_name + "_" + binary_filename.replace(binary_file_extension, "zip")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(file_path, arcname=os.path.basename(file_path))
    return zip_path

def deleteFileFromBin(jm19_bin_path):
    files = os.listdir(jm19_bin_path)
    for file in files:
        if file in FILE_TO_REMOVE:
            os.remove(os.path.join(jm19_bin_path, file))


if __name__ == '__main__':
    #generateTXT("/Prove/Bertazzini/SOCIAL_DATASET", "/Prove/Bertazzini")
    extractVideoData(args.video_path, args.social, args.video_data_output_path, args.jm19_bin_path, args.generate_h264)
