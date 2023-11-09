import argparse
import os
import pickle
import statistics
from time import time

import h5py
import numpy as np
import re

from colorama import Fore
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut

parser = argparse.ArgumentParser()
parser.add_argument('--results_save_path', type=str, help='location of the folder to save the results (i.e. the confusion matrices)')
parser.add_argument('--scenario', type=str, help='base, ffmpeg or avidemux')
parser.add_argument('--inference', type=bool, default=False, help='if True, it will use the corresponding saved model')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--dct_luma', type=bool)
parser.add_argument('--dct_chroma', type=bool)
parser.add_argument('--mb', type=bool)
parser.add_argument('--iframes', type=bool, default=False)
parser.add_argument('--qp', type=bool)
parser.add_argument('--h5_dataset_path', type=str, default="/Prove/Bertazzini/HDF5_DATASETS")
parser.add_argument('--models_save_path', type=str)
args = parser.parse_args()

SOCIAL = ['Facebook', 'Instagram', 'Twitter', 'Youtube', 'native']
DEVICES = ['M30', 'M23', 'M13', 'D41', 'F02', 'M10', 'D42', 'M27', 'F04', 'M29', 'D38', 'M16', 'D37', 'M28', 'M12', 'M32', 'D43', 'M18', 'M01', 'D39', 'F03', 'M00', 'F01', 'D36', 'M17', 'D40', 'M11', 'M19']
BRANDS = ['Apple', 'Asus', 'Google', 'Huawei', 'Lenovo', 'LG', 'Motorola', 'Nokia', 'Samsung', 'Wiko', 'Xiaomi']
FIRMWARE = ['Android_10', 'Android_11', 'Android_5', 'Android_6', 'Android_7', 'Android_8', 'iOS_11', 'iOS_9', 'Unk']

def fit_dct_features(feature_matrix, labels_array):
    clf = RandomForestClassifier(100)
    clf.fit(feature_matrix, labels_array)
    return clf
def loadDataset(scenario, dataset_name, test_idx, luma, chroma, mb, qp, iframes):
    X_train, X_test = [], []
    y_train, y_test = [], []
    X, y = [], []

    scenario_file = h5py.File(args.h5_dataset_path + f"/{scenario}_{dataset_name}.hdf5", "r")

    if luma or chroma:
        if luma and not chroma:
            for idx, feature_matrix in enumerate(scenario_file["dct"][:, :, :18]):
                if scenario_file["devices"][idx] == test_idx:
                    X_test.append(feature_matrix)
                    y_test.append(scenario_file["labels"][idx])
                else:
                    X_train.append(feature_matrix)
                    y_train.append(scenario_file['labels'][idx])

        elif not luma and chroma:
            for idx, feature_matrix in enumerate(scenario_file["dct"][:, :, 18:]):
                if scenario_file["devices"][idx] == test_idx:
                    X_test.append(feature_matrix)
                    y_test.append(scenario_file["labels"][idx])
                else:
                    X_train.append(feature_matrix)
                    y_train.append(scenario_file['labels'][idx])

        elif luma and chroma:
            for idx, feature_matrix in enumerate(scenario_file["dct"][:]):
                if scenario_file["devices"][idx] == test_idx:
                    X_test.append(feature_matrix)
                    y_test.append(scenario_file["labels"][idx])
                else:
                    X_train.append(feature_matrix)
                    y_train.append(scenario_file['labels'][idx])

    if mb:
        for idx, feature_matrix in enumerate(scenario_file["macroblocks"][:]):
            if scenario_file["devices"][idx] == test_idx:
                X_test.append(feature_matrix)
                y_test.append(scenario_file["labels"][idx])
            else:
                X_train.append(feature_matrix)
                y_train.append(scenario_file['labels'][idx])
    if qp:
        for idx, feature_matrix in enumerate(scenario_file["qps"][:]):
            if scenario_file["devices"][idx] == test_idx:
                X_test.append(feature_matrix[:2])
                y_test.append(scenario_file["labels"][idx])
            else:
                X_train.append(feature_matrix[:2])
                y_train.append(scenario_file['labels'][idx])

    if iframes:
        for idx, feature_matrix in enumerate(scenario_file["iframes"][:]):
            if scenario_file["devices"][idx] == test_idx:
                X_test.append(feature_matrix)
                y_test.append(scenario_file["labels"][idx])
            else:
                X_train.append(feature_matrix)
                y_train.append(scenario_file['labels'][idx])

    return X_train, X_test, y_train, y_test
def trainRF(results_path, models_save_path, scenario, dataset_name, normalization, luma, chroma, mb, qp, iframes, inference, cmap):
    start = time()
    y_preds, y_trues = [], []
    cms, acc, bal_acc = [], [], []
    features_length = 0

    if not inference:
        for q in range(10):
            print(Fore.LIGHTBLUE_EX + f"ITERATION {q+1}/10" + Fore.RESET)
            classifiers = []
            loo = LeaveOneOut()
            for i, (train_idx, test_idx) in enumerate(loo.split(DEVICES)):
                classifiers_name = f"SOCIALID_RF"
                if luma or chroma:
                    if luma and not chroma:
                        X_train, X_test, y_train, y_test = loadDataset(scenario, dataset_name, test_idx, luma=True, chroma=False, mb=False, qp=False, iframes=False)
                        classifiers_name += "_luma"
                    elif not luma and chroma:
                        X_train, X_test, y_train, y_test = loadDataset(scenario, dataset_name, test_idx, luma=True, chroma=True, mb=False, qp=False, iframes=False)
                        classifiers_name += "_chroma"
                    elif luma and chroma:
                        X_train, X_test, y_train, y_test = loadDataset(scenario, dataset_name, test_idx, luma=True, chroma=True, mb=False, qp=False, iframes=False)
                        classifiers_name += "_luma_chroma"
                    else:
                        raise

                    X_train = np.array(X_train)
                    X_test = np.array(X_test)

                    if normalization:
                        for k in range(X_train.shape[0]):
                            column_sum = np.sum(X_train[k], axis=0)
                            column_sum[column_sum == 0] = 1
                            X_train[k] = X_train[k] / column_sum
                        X_train = X_train.reshape(X_train.shape[0], -1)

                        for k in range(X_test.shape[0]):
                            column_sum = np.sum(X_test[k], axis=0)
                            column_sum[column_sum == 0] = 1
                            X_test[k] = X_test[k] / column_sum
                        X_test = X_test.reshape(X_test.shape[0], -1)

                if mb:
                    classifiers_name += "_mb"
                    if luma or chroma:
                        mb_train, mb_test, _, _ = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=True, qp=False, iframes=False)
                        mb_train = np.array(mb_train)
                        mb_train = mb_train.reshape(mb_train.shape[0], -1)
                        mb_test = np.array(mb_test)
                        mb_test = mb_test.reshape(mb_test.shape[0], -1)
                        X_train = np.concatenate((X_train, mb_train), axis=1)
                        X_test = np.concatenate((X_test, mb_test), axis=1)

                    else:
                        X_train, X_test, y_train, y_test = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=True, qp=False, iframes=False)
                        X_train = np.array(X_train)
                        X_test = np.array(X_test)
                        X_train = X_train.reshape(X_train.shape[0], -1)
                        X_test = X_test.reshape(X_test.shape[0], -1)

                if qp:
                    classifiers_name += "_qp"
                    if luma or chroma or mb:
                        qp_train, qp_test, _, _ = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=True, iframes=False)
                        qp_train = np.array(qp_train)
                        qp_train = qp_train.reshape(qp_train.shape[0], -1)
                        qp_test = np.array(qp_test)
                        qp_test = qp_test.reshape(qp_test.shape[0], -1)
                        X_train = np.concatenate((X_train, qp_train), axis=1)
                        X_test = np.concatenate((X_test, qp_test), axis=1)

                    else:
                        X_train, X_test, y_train, y_test = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=True, iframes=False)
                        X_train = np.array(X_train)
                        X_test = np.array(X_test)
                        X_train = X_train.reshape(X_train.shape[0], -1)
                        X_test = X_test.reshape(X_test.shape[0], -1)


                if iframes:
                    classifiers_name += '_iframes'
                    if luma or chroma or mb or qp:
                        iframes_train, iframes_test, _, _ = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=False, iframes=True)
                        iframes_train = np.array(iframes_train)
                        iframes_train = iframes_train.reshape(iframes_train.shape[0], -1)
                        iframes_test = np.array(iframes_test)
                        iframes_test = iframes_test.reshape(iframes_test.shape[0], -1)
                        X_train = np.concatenate((X_train, iframes_train), axis=1)
                        X_test = np.concatenate((X_test, iframes_test), axis=1)

                    else:
                        X_train, X_test, y_train, y_test = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=False, iframes=True)
                        X_train = np.array(X_train)
                        X_test = np.array(X_test)
                        X_train = X_train.reshape(X_train.shape[0], -1)
                        X_test = X_test.reshape(X_test.shape[0], -1)


                print(f'GRID SEARCH {i+1}/{len(DEVICES)}')
                clf = fit_dct_features(X_train, y_train)

                classifiers.append(clf)

                y_pred = clf.predict(X_test)

                y_preds += list(y_pred)
                y_trues += list(y_test)

                print("ACC: ", accuracy_score(y_pred, y_test))


            with open(f'{models_save_path}/{classifiers_name}_{q}.pkl', 'wb') as f:
                pickle.dump(classifiers, f)

    else:
        for q in range(10):
            print(Fore.LIGHTBLUE_EX + f"Iteration {q+1}/10" + Fore.RESET)
            loo = LeaveOneOut()
            for i, (train_idx, test_idx) in enumerate(loo.split(DEVICES)):
                print(f'DEVICE {i + 1}/{len(DEVICES)}')
                classifiers_name = f"SOCIALID_RF"
                if luma or chroma:
                    if luma and not chroma:
                        features_length += 2000*18
                        _, X_test,_, y_test = loadDataset(scenario, dataset_name, test_idx, luma=True, chroma=False,mb=False, qp=False, iframes=False)
                        classifiers_name += "_luma"
                    elif not luma and chroma:
                        features_length += 2000 * 36
                        _, X_test, _, y_test = loadDataset(scenario, dataset_name, test_idx, luma=True, chroma=True, mb=False, qp=False, iframes=False)
                        classifiers_name += "_chroma"
                    elif luma and chroma:
                        features_length += 2000 * 54
                        _, X_test, _, y_test = loadDataset(scenario, dataset_name, test_idx, luma=True, chroma=True, mb=False, qp=False, iframes=False)
                        classifiers_name += "_luma_chroma"
                    else:
                        raise

                    X_test = np.array(X_test)

                    if normalization:
                        for k in range(X_test.shape[0]):
                            column_sum = np.sum(X_test[k], axis=0)
                            column_sum[column_sum == 0] = 1
                            X_test[k] = X_test[k] / column_sum
                        X_test = X_test.reshape(X_test.shape[0], -1)

                if mb:
                    classifiers_name += "_mb"
                    features_length += 21
                    if luma or chroma:
                        _, mb_test, _, _ = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=True, qp=False, iframes=False)
                        mb_test = np.array(mb_test)
                        mb_test = mb_test.reshape(mb_test.shape[0], -1)
                        X_test = np.concatenate((X_test, mb_test), axis=1)

                    else:
                        _, X_test, _, y_test = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=True, qp=False, iframes=False)
                        X_test = np.array(X_test)
                        X_test = X_test.reshape(X_test.shape[0], -1)

                if qp:
                    classifiers_name += "_qp"
                    features_length += 104
                    if luma or chroma or mb:
                        _, qp_test, _, _ = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=True, iframes=False)
                        qp_test = np.array(qp_test)
                        qp_test = qp_test.reshape(qp_test.shape[0], -1)
                        X_test = np.concatenate((X_test, qp_test), axis=1)

                    else:
                        _, X_test, _, y_test = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=True, iframes=False)
                        X_test = np.array(X_test)
                        X_test = X_test.reshape(X_test.shape[0], -1)

                if iframes:
                    classifiers_name += '_iframes'
                    features_length += 1
                    if luma or chroma or mb or qp:
                        _, iframes_test, _, _ = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False,  qp=False, iframes=True)
                        iframes_test = np.array(iframes_test)
                        iframes_test = iframes_test.reshape(iframes_test.shape[0], -1)
                        X_test = np.concatenate((X_test, iframes_test), axis=1)

                    else:
                        _, X_test, _, y_test = loadDataset(scenario, dataset_name, test_idx, luma=False, chroma=False, mb=False, qp=False, iframes=True)
                        X_test = np.array(X_test)

                with open(f'{models_save_path}/{classifiers_name}_{q}.pkl', 'rb') as f:
                    classifiers = pickle.load(f)

                # importances = np.zeros((len(classifiers), features_length))
                # print("Computing importances...")
                # for i in range(len(classifiers)):
                #     importances[i] = classifiers[i].feature_importances_
                # importances = np.mean(importances, axis=0)
                # sorted_importances = sorted(importances, reverse=True)
                # sorted_indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
                #
                # std = []
                # # print("Computing standard deviations...")
                # # for classifier in classifiers:
                # #     std.append(np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0))
                # # std = np.mean(std, axis=0)
                # # sorted_std = [std[i] for i in sorted_indices]
                #
                # print("Plotting...")
                # forest_importances = pd.Series(sorted_importances[:20])
                # fig, ax = plt.subplots(figsize=(19,10))
                # labels = []
                # for idx in sorted_indices[:20]:
                #     if idx == 108021:
                #         labels.append('I-frames')
                #     elif idx >= 108014 and idx <= 108020:
                #         labels.append('MB_B-frames')
                #     elif idx >= 108007 and idx <= 108013:
                #         labels.append('MB_P-frames')
                #     elif idx >= 108000 and idx <= 108006:
                #         labels.append('MB_I-frames')
                #     elif idx >= 72000 and idx <= 107999:
                #         tmp_idx = idx - 72000
                #         if tmp_idx < 18000:
                #             type = "8x8"
                #             ac_index = tmp_idx // 2000
                #         else:
                #             type = "16x16_4x4"
                #             ac_index = (tmp_idx-18000) // 2000
                #         labels.append(f'Cr_AC_{str(ac_index)}_{type}')
                #     elif idx >= 36000 and idx <= 71999:
                #         tmp_idx = idx - 36000
                #         if tmp_idx < 18000:
                #             type = "8x8"
                #             ac_index = tmp_idx // 2000
                #         else:
                #             type = "16x16_4x4"
                #             ac_index = (tmp_idx-18000) // 2000
                #         labels.append(f'Cb_AC_{str(ac_index)}_{type}')
                #     else:
                #         if idx < 18000:
                #             type = "8x8"
                #             ac_index = idx // 2000
                #         else:
                #             type = "16x16_4x4"
                #             ac_index = (idx - 18000) // 2000
                #
                #         labels.append(f'Luma_AC_{str(ac_index)}_{type}')
                #
                # colors = []
                # patches = []
                # for label in labels:
                #     if label == 'I-frames':
                #         colors.append('tab:pink')
                #     elif "MB" in label:
                #         colors.append('tab:cyan')
                #     elif 'Luma' in label:
                #         colors.append('tab:blue')
                #     else:
                #         colors.append('tab:purple')
                #
                # for color in ['tab:pink', 'tab:cyan', 'tab:blue', 'tab:purple']:
                #     patches.append(mpatches.Rectangle((0, 0), 1, 1, fc=color))
                #
                # forest_importances.plot.bar(ax=ax, color=colors) #yerr=sorted_std[:100]
                # legend_labels = ['I-frames', 'Macroblock types', 'DCT Luma', 'DCT Chroma']
                # ax.legend(patches, legend_labels, fontsize=26)
                # ax.set_xticks(range(len(labels)))
                # ax.set_xticklabels(labels, fontsize=22)
                # ax.set_title("Feature importances", fontweight='bold', fontsize=28)
                # ax.set_ylim(importances[0]+0.05*importances[0])
                # ax.set_yticklabels(['0.000', '0.001', '0.002', '0.003', '0.004', '0.005'], fontsize=22)
                # ax.set_ylabel("Mean decrease in impurity", fontsize=22)
                # fig.tight_layout()
                #
                # plt.savefig(f"{results_path}/{classifiers_name.replace('SOCIALID_RF_', '')}_features_importances.pdf")
                # plt.close()


                y_pred = classifiers[i].predict(X_test)

                y_preds += list(y_pred)
                y_trues += list(y_test)


            cm = confusion_matrix(y_trues, y_preds, normalize='true')
            cms.append(cm)
            acc.append(accuracy_score(y_trues, y_preds))
            bal_acc.append(balanced_accuracy_score(y_trues, y_preds))

        cm = np.mean(cms, axis=0)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SOCIAL)
        disp.plot(cmap=cmap)

        title = f'{scenario.upper()} scenario - MeanAcc: {round(statistics.mean(acc), 2)}, σ: {round(statistics.stdev(acc), 3)} (MeanBalAcc: {round(statistics.mean(bal_acc), 2)}, σ: {round(statistics.stdev(bal_acc), 3)})'

        plt.title(title, fontdict={'size': 10})

        fig_name = f'cm_{scenario}_scenario'

        plt.savefig(f"{results_path}/{fig_name}_{classifiers_name.replace('RF_', '')}.pdf")

    print(f"Finish in around {round((time() - start) / 60)} minutes")

def findAllIndicesString(to_find, string):
    indices_object = re.finditer(pattern=to_find, string=string)
    return [index.start() for index in indices_object]


if __name__ == '__main__':
    RESULTS_PATH = "/Prove/Bertazzini/RESULTS"
    os.makedirs(RESULTS_PATH, exist_ok=True)

    SCENARIO = "base"
    NORMALIZATION = True
    INFERENCE = True
    CMAP = 'RdPu'

    DATASET_NAME = "social_identification"

    MODELS_SAVE_PATH = "/Prove/Bertazzini/MODELS/SOCIAL_IDENTIFICATION"
    os.makedirs(MODELS_SAVE_PATH, exist_ok=True)

    file = open('/data/lesc/staff/bertazzini/VideoData/features_combination.txt', 'r')
    combinations = file.readlines()

    for combination in combinations:
        print(Fore.LIGHTGREEN_EX + f"Combination {combination[:combination.find('n')]}" + Fore.RESET)
        indices = findAllIndicesString(',', combination)
        LUMA = True if combination[:indices[0]] == 'True' else False
        CHROMA = True if combination[indices[0]+2:indices[1]] == 'True' else False
        # MB = True if combination[indices[1]+2:indices[2]] == 'True' else False
        QP = True if combination[indices[1]+2:combination.find("\n")] == 'True' else False

        MB = False
        IFRAMES = False

        trainRF(RESULTS_PATH, MODELS_SAVE_PATH, SCENARIO, DATASET_NAME, NORMALIZATION, LUMA, CHROMA, MB, QP, IFRAMES, INFERENCE, CMAP)