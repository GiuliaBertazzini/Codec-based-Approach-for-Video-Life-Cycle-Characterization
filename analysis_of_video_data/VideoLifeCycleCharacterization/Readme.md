# Video life-cycle characterization

## Requirements
* **JM19.0 Software**, that you have to download and compile from the following [link](https://git.lesc.dinfo.unifi.it/bertazzini/JM19)
* **Python 3.9** or more

## What these scripts do
Thanks to a modified version of JM Reference software, you can extract all the relevant information from an encoded H.264 bitstrem, optimizing in the mean while memory storage and decoding time. Then, you can analyze the extracted features to determine which of them are potentially discriminating in the considered forensic scenario (i.e., the identification of the originating Social Network).

## Video data extraction
To simply extract video data from a video dataset (which contains video from different Socials), run the following command, from DataExtraction directory:

```bash
python extract_video_data_social.py --video_path --social --video_data_output_path --generate_h264 --jm19_bin_path 
```
where *video_path* specifies the path of the video to process, *social* determines the social of the considered video, *video_data_output_path* specifies the location of the output directory for H.264 generated video and binary output files, *generate_h264* a boolean that if set to True generates the .h264 corresponding files, otherwise it looks for this file in the *video_data_output_path* (default False) and *jm19_bin_path* specifies the directory containing the ldecod.exe and decode.cfg files. 

You can generate with this script a txt file containing the video paths and the corresponding social to launch the program with slurm (you can specify the parameters of interest in parallel_extract.sh file).

After that, the script will produce for each social a directory named *BINARY_OUTPUTS*, in which you can find the zipped binary files produced by JM. These files need to be processed by a RUST script, which you can launch with the following command:

```bash
python extract_histogram_from_binary_files.py --file_zip_path --social --video_data_output_path
```

where *file_zip_path* specify the path of a binary file produced by JM and *social* the corresponding social; *video_data_output_path* determines the location of the directory containing the social as subdirectories (in each of which is included the BINARY_OUTPUTS dir). 

Even in this case, the command can be launch with slurm properly modifing the .sh file.

After that, the script will produce a new directory for each social, named "HISTOGRAMS" in which you can find the json files of the DCT histograms, macroblocks distribution, QP histogram, and so on.  

The final step requires to better organize the extracted data in order to analyze them later. To do this, use the following command: 
```bash
python organize_data.py --json_histograms_path --ac_matrices_save_path --h5_dataset_save_path --dataset_name 
```
where *json_histograms_path* specify the path to the directory containg the social subdirectories, in each of which has been created the "HISTOGRAMS" directory; *ac_matrices_save_path* if specified, determine the location of the folder to save the AC matrices of each social, otherwise it will be created a directory named "AC_MATRICES" in the same location of "HISTOGRAMS" directory; *h5_dataset_save_path* identify the path of the folder to save the hdf5 dataset; *dataset_name* specifies the name you want to give the dataset.

After that, the script will produce a new directory for each social, named "AC_MATRICES", containing a pickle file for each video in which have been stored the ac matrices for each video (three 2000x18 matrices, one for dct luma and two for dct chroma). 
Moreover, three .hdf5 files will be produced (for base, ffmpeg and avidemux scenario). With these file created, you can procede to the analysis of your data. 

## Video data analysis
With RandomForest.py script you can analyze the previously extracted features, using the following command: 

```bash
python RandomForest.py 
```
In this case, you have to specify a fairly large number of parameters:
* results_path: determines the location of the directory to save the plot of the confusion matrices (.pdf files)
* scenario: the scenario of interest for the considered experiment (base, ffmpeg or avidemux)
* inference: if False, it will train and save the models, otherwise it will inference with the saved models
* dataset_name: the name of the dataset you want to use fore the experiments 
* models_save_path: the location of the directory to save the trained models
* luma, chroma, mb and qp: boolean parameters that if set to True the corresponding parameter will be considered in the experiments, otherwise not