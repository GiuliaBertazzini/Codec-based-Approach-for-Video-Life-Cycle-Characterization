import argparse
import csv
import os
import pickle

import h5py
import numpy as np
import json
from read_video_data_optimized import zigzag

parser = argparse.ArgumentParser()
parser.add_argument('--json_histograms_path', type=str, default='/Prove/Bertazzini/SOCIAL_DATASET', help='location of the social folders containing json histograms')
parser.add_argument('--ac_matrices_save_path', type=str, default='/Prove/Bertazzini/SOCIAL_DATASET', help='location of the folder to save AC matrices')
parser.add_argument('--h5_dataset_save_path', type=str, default='/Prove/Bertazzini/HDF5_DATASETS', help='location of the folder to save H5 dataset')
parser.add_argument('--dataset_name', type=str, help='name of H5 dataset')
args = parser.parse_args()

SOCIAL = ['Facebook', 'Instagram', 'Twitter', 'Youtube', 'native']
DEVICES = ['M30', 'M23', 'M13', 'D41', 'F02', 'M10', 'D42', 'M27', 'F04', 'M29', 'D38', 'M16', 'D37', 'M28', 'M12', 'M32', 'D43', 'M18', 'M01', 'D39', 'F03', 'M00', 'F01', 'D36', 'M17', 'D40', 'M11', 'M19']
BRANDS = ['Apple', 'Asus', 'Google', 'Huawei', 'Lenovo', 'LG', 'Motorola', 'Nokia', 'Samsung', 'Wiko', 'Xiaomi']
FIRMWARE = ['Android_10', 'Android_11', 'Android_5', 'Android_6', 'Android_7', 'Android_8', 'iOS_11', 'iOS_9', 'Unk']


def generateACMatrixFromHistograms(json_histograms_path, ac_matrices_save_path=None):
    for social in os.listdir(json_histograms_path):

        video_names = os.listdir(f"{json_histograms_path}/{social}/HISTOGRAMS")

        for video_name in video_names:
            print(f"ANALYSIS OF {video_name}...")
            with open(f"{json_histograms_path}/{social}/HISTOGRAMS/{video_name}", 'r') as file:
                data_histograms = json.load(file)

            i_frames = data_histograms['i_frames_count']

            # ORGANIZE DCT OF LUMA COMPONENT
            new_dict = {'hist_8x8': {}, 'hist_16x16_4x4': {}}

            list_of_indices = [i for i in range(len(data_histograms['global_histograms']['hist_8x8']))]
            list_of_indices = np.array(list_of_indices).reshape((8, 8))
            indices_zscanned_8x8 = zigzag(list_of_indices)
            for ac_idx, idx in enumerate(list(indices_zscanned_8x8[1:10])):
                new_dict['hist_8x8'][ac_idx] = data_histograms['global_histograms']['hist_8x8'][idx]

            list_of_indices = [i for i in range(len(data_histograms['global_histograms']['hist_4x4']))]
            list_of_indices = np.array(list_of_indices).reshape((4, 4))
            indices_zscanned_4x4_16x16 = zigzag(list_of_indices)

            for ac_idx, idx in enumerate(list(indices_zscanned_4x4_16x16[1:10])):
                new_dict['hist_16x16_4x4'][ac_idx] = data_histograms['global_histograms']['hist_4x4'][idx]

                for key in data_histograms['global_histograms']['hist_16x16'][idx].keys():
                    if key in new_dict['hist_16x16_4x4'][ac_idx].keys():
                        new_dict['hist_16x16_4x4'][ac_idx][key] += data_histograms['global_histograms']['hist_16x16'][idx][key]
                    else:
                        new_dict['hist_16x16_4x4'][ac_idx][key] = data_histograms['global_histograms']['hist_16x16'][idx][key]

            # ORGANIZE DCT FOR CHROMA BLUE COMPONENT
            new_dict_cb = {'hist_8x8': {}, 'hist_16x16_4x4': {}}

            for ac_idx, idx in enumerate(list(indices_zscanned_4x4_16x16[1:10])):
                new_dict_cb['hist_8x8'][ac_idx] = data_histograms['global_histograms']['hist_8x8_cb'][idx]
                new_dict_cb['hist_16x16_4x4'][ac_idx] = data_histograms['global_histograms']['hist_4x4_cb'][idx]

                for key in data_histograms['global_histograms']['hist_16x16_cb'][idx].keys():
                    if key in new_dict_cb['hist_16x16_4x4'][ac_idx].keys():
                        new_dict_cb['hist_16x16_4x4'][ac_idx][key] += data_histograms['global_histograms']['hist_16x16_cb'][idx][key]
                    else:
                        new_dict_cb['hist_16x16_4x4'][ac_idx][key] = data_histograms['global_histograms']['hist_16x16_cb'][idx][key]

            # ORGANIZE DCT FOR CHROMA RED COMPONENT
            new_dict_cr = {'hist_8x8': {}, 'hist_16x16_4x4': {}}

            for ac_idx, idx in enumerate(list(indices_zscanned_4x4_16x16[1:10])):
                new_dict_cr['hist_8x8'][ac_idx] = data_histograms['global_histograms']['hist_8x8_cr'][idx]
                new_dict_cr['hist_16x16_4x4'][ac_idx] = data_histograms['global_histograms']['hist_4x4_cr'][idx]

                for key in data_histograms['global_histograms']['hist_16x16_cr'][idx].keys():
                    if key in new_dict_cr['hist_16x16_4x4'][ac_idx].keys():
                        new_dict_cr['hist_16x16_4x4'][ac_idx][key] += data_histograms['global_histograms']['hist_16x16_cr'][idx][key]
                    else:
                        new_dict_cr['hist_16x16_4x4'][ac_idx][key] = data_histograms['global_histograms']['hist_16x16_cr'][idx][key]


            # Load Data
            num_bin = 2001
            min_dct = -1000
            max_dct = 1000
            AC_histograms_matrix_8x8 = np.zeros((num_bin, 9),dtype=int)
            AC_histograms_matrix_16x16_4x4 = np.zeros((num_bin, 9), dtype=int)
            AC_histograms_matrix_8x8_cb = np.zeros((num_bin, 9), dtype=int)
            AC_histograms_matrix_16x16_4x4_cb = np.zeros((num_bin, 9), dtype=int)
            AC_histograms_matrix_8x8_cr = np.zeros((num_bin, 9), dtype=int)
            AC_histograms_matrix_16x16_4x4_cr = np.zeros((num_bin, 9), dtype=int)

            # LUMA COMPONENT
            data = new_dict['hist_8x8']
            for ac in range(9):
                AC_histogram = np.zeros((num_bin), dtype=int)
                for dct_coeff in data[ac].keys():
                    if int(dct_coeff) >= min_dct and int(dct_coeff) <= max_dct:
                        AC_histogram[num_bin // 2 + int(dct_coeff)] = data[ac][dct_coeff] / (i_frames)
                AC_histograms_matrix_8x8[:, ac] += AC_histogram


            data = new_dict['hist_16x16_4x4']
            for ac in range(9):
                AC_histogram = np.zeros((num_bin), dtype=int)
                for dct_coeff in data[ac].keys():
                    # discard dct coefficients which exceed the number of bins
                    if int(dct_coeff) >= min_dct and int(dct_coeff) <= max_dct:
                        AC_histogram[num_bin // 2 + int(dct_coeff)] = data[ac][dct_coeff] / (i_frames)
                AC_histograms_matrix_16x16_4x4[:, ac] += AC_histogram

            # CHROMA BLUE COMPONENT

            data = new_dict_cb['hist_8x8']
            for ac in range(9):
                AC_histogram = np.zeros((num_bin), dtype=int)
                for dct_coeff in data[ac].keys():
                    if int(dct_coeff) >= min_dct and int(dct_coeff) <= max_dct:
                        AC_histogram[num_bin // 2 + int(dct_coeff)] = data[ac][dct_coeff]  / (i_frames)
                AC_histograms_matrix_8x8_cb[:, ac] += AC_histogram

            data = new_dict_cb['hist_16x16_4x4']
            for ac in range(9):
                AC_histogram = np.zeros((num_bin), dtype=int)
                for dct_coeff in data[ac].keys():
                    # discard dct coefficients which exceed the number of bins
                    if int(dct_coeff) >= min_dct and int(dct_coeff) <= max_dct:
                        AC_histogram[num_bin // 2 + int(dct_coeff)] = data[ac][dct_coeff] / (i_frames)
                AC_histograms_matrix_16x16_4x4_cb[:, ac] += AC_histogram

            # CHROMA RED COMPONENT

            data = new_dict_cr['hist_8x8']
            for ac in range(9):
                AC_histogram = np.zeros((num_bin), dtype=int)
                for dct_coeff in data[ac].keys():
                    if int(dct_coeff) >= min_dct and int(dct_coeff) <= max_dct:
                        AC_histogram[num_bin // 2 + int(dct_coeff)] = data[ac][dct_coeff]  / (i_frames)
                AC_histograms_matrix_8x8_cr[:, ac] += AC_histogram

            data = new_dict_cr['hist_16x16_4x4']
            for ac in range(9):
                AC_histogram = np.zeros((num_bin), dtype=int)
                for dct_coeff in data[ac].keys():
                    # discard dct coefficients which exceed the number of bins
                    if int(dct_coeff) >= min_dct and int(dct_coeff) <= max_dct:
                        AC_histogram[num_bin // 2 + int(dct_coeff)] = data[ac][dct_coeff]  / (i_frames)
                AC_histograms_matrix_16x16_4x4_cr[:, ac] += AC_histogram

            # AC MATRIX WITH DCT LUMA, DCT CB AND DCT CR (IN THIS ORDER)
            AC_histrogram_concat = np.concatenate((AC_histograms_matrix_8x8, AC_histograms_matrix_16x16_4x4, AC_histograms_matrix_8x8_cb, AC_histograms_matrix_16x16_4x4_cb, AC_histograms_matrix_8x8_cr, AC_histograms_matrix_16x16_4x4_cr), axis=1)

            if ac_matrices_save_path is not None:
                matrices_save_dir = f"{ac_matrices_save_path}/{social}"
            else:
                matrices_save_dir = json_histograms_path + f"/{social}/AC_MATRICES_PROVA"
            os.makedirs(matrices_save_dir, exist_ok=True)

            with open(f"{matrices_save_dir}/{video_name.replace('_histogram.json', '')}_AC_matrix.pkl", 'wb') as f:
                pickle.dump(AC_histrogram_concat, f)


def createH5Dataset(histograms_path, ac_matrices_path, h5_dataset_save_path, dataset_name):
    for scenario in ['base', 'ffmpeg', 'avidemux']:
        print(f"Generating dataset for {scenario} scenario")
        dataset_path = os.path.join(h5_dataset_save_path, f"{scenario}_{dataset_name}.hdf5")
        dct, labels, mb, iframes, qps, devices, dev_social = readDCTDataset(histograms_path, ac_matrices_path, scenario=scenario)
        generateH5Dataset(dataset_path, dct, f"dct", "f8")
        generateH5Dataset(dataset_path, labels, f"labels", "i")
        generateH5Dataset(dataset_path, mb, 'macroblocks', "f8")
        generateH5Dataset(dataset_path, qps, 'qps', "f8")
        generateH5Dataset(dataset_path, iframes, 'iframes', "f8")
        generateH5Dataset(dataset_path, devices, 'devices', "i")

def generateH5Dataset(dataset_location, data_matrix, data_name, data_type):
    with h5py.File(dataset_location, "a") as f:
        f.create_dataset(data_name, data_matrix.shape, dtype=data_type, data=data_matrix)

def readDCTDataset(histograms_path, ac_matrices_path, scenario='base'):
    X, y, mbs, iframes, qps, devices, dev_social  = [], [], [], [], [], [], []

    if scenario == 'base':
        SOCIAL_TO_LOAD = SOCIAL
    elif scenario == 'ffmpeg' or scenario == 'avidemux':
        SOCIAL_TO_LOAD = [social + f'-{scenario}' for social in SOCIAL]
    else:
        raise

    for social in SOCIAL_TO_LOAD:
        ac_mat_social_path = ac_matrices_path + f"/{social}/AC_MATRICES_PROVA"
        list_of_files = os.listdir(ac_mat_social_path)

        for file in list_of_files:
            devices.append(DEVICES.index(file[:3]))

            if scenario == 'ffmpeg' or scenario == 'avidemux':
                y.append(SOCIAL.index(social.replace(f"-{scenario}", "")))
            else:
                y.append(SOCIAL.index(social))

            video_name = file.replace("_AC_matrix.pkl", "")

            # ORGANIZE DCT

            with open(f"{ac_mat_social_path}/{file}", "rb") as f:
                dct_data = pickle.load(f)

            data1 = dct_data[0:1000]
            data2 = dct_data[1001:2001]

            data = np.vstack([data1, data2])

            X.append(data)

            with open(f"{histograms_path}/{social}/HISTOGRAMS/{video_name}_histogram.json", 'r') as f:
                data = json.load(f)

            frames_types = data['frames_types']
            i = frames_types.count('I')
            p = frames_types.count('P')
            b = frames_types.count('B')

            # GET MACROBLOCKS DISTRIBUTION

            mb_vector = np.zeros((3, 7))
            macroblocks = data['macroblocks_counts']

            for frame_id, mb_types in enumerate(macroblocks):
                for id, value in enumerate(mb_types.values()):
                    if frames_types[frame_id] == 'I':
                        mb_vector[0][id] += value
                    elif frames_types[frame_id] == 'P':
                        mb_vector[1][id] += value
                    elif frames_types[frame_id] == 'B':
                        mb_vector[2][id] += value
                    else:
                        raise
            if i>0:
                mb_vector[0] = mb_vector[0]/i
            if p>0:
                mb_vector[1] = mb_vector[1]/p
            if b>0:
                mb_vector[2] = mb_vector[2]/b

            mbs.append(mb_vector)

            # GET QP DISTRIBUTION

            qp = np.zeros((3, 52))
            for _, value in data['qp_counts_y'].items():
                qp[0] += np.array(value)
            qp[0] = [qp[0][i]/sum(qp[0]) for i in range(len(qp[0]))]
            for _, value in data['qp_counts_u'].items():
                qp[1] += np.array(value)
            qp[1] = [qp[1][i]/sum(qp[1]) for i in range(len(qp[1]))]
            for _, value in data['qp_counts_v'].items():
                qp[2] += np.array(value)
            qp[2] = [qp[2][i]/sum(qp[2]) for i in range(len(qp[2]))]
            qps.append(qp)

            iframe_perc = data['i_frames_count']/data['frames_count']
            iframes.append(iframe_perc)

            dev_social.append(f"{file[:3]}_{social}")

    X = np.array(X)
    y = np.array(y)
    mbs = np.array(mbs)
    qps = np.array(qps)
    iframes = np.array(iframes)
    devices = np.array(devices)
    dev_social = np.array(dev_social)
    return X, y, mbs, iframes, qps, devices, dev_social

if __name__ == '__main__':
    generateACMatrixFromHistograms(args.json_histograms_path, args.ac_matrices_save_path)
    createH5Dataset(args.json_histograms_path, args.ac_matrices_save_path, args.h5_dataset_save_path, "social_identification_prova")
