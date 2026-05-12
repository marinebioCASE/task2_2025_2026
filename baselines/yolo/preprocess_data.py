import json
import shutil
import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F_general
import torchaudio
import torchaudio.functional as F
from PIL import Image
from tqdm import tqdm
import datetime

pd.set_option('future.no_silent_downcasting', True)
random.seed(42)


class YOLODataset:
    def __init__(self, config, path_to_dataset):
        # Spectrogram settings
        self.duration = config['duration']
        self.overlap = config['overlap']  # overlap of the chunks in %
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.color = config['color']

        self.nfft = config['nfft']
        self.win_len = config['win_len']
        self.hop_len = config['hop_len']
        self.win_overlap = self.win_len - self.hop_len

        # Get the color map by name:
        #self.cmap = plt.get_cmap(config['cmap'])

        # Path to dataset
        self.path_to_dataset = pathlib.Path(path_to_dataset)
        self.wavs_folder = self.path_to_dataset.joinpath('raw')
        self.annotations_folder = self.path_to_dataset.joinpath('annotations')
        self.images_folder = self.path_to_dataset.joinpath('images')
        self.labels_folder = self.path_to_dataset.joinpath('labels')
        if not self.images_folder.exists():
            os.mkdir(self.images_folder)
            os.mkdir(self.labels_folder)
            os.mkdir(self.labels_folder.joinpath('backgrounds'))

        self.F_MIN = 0
        self.blocksize = int(self.duration * self.desired_fs)
        self.config = config

    def __setitem__(self, key, value):
        if key in self.config.keys():
            self.config[key] = value
        self.__dict__[key] = value

    def save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def create_train_dataset(self, class_encoding):
        indices_per_deployment = self.convert_challenge_annotations_to_yolo(class_encoding=class_encoding)
        selected_samples = self.select_background_labels(indices_per_deployment)
        self.create_spectrograms(selected_samples=selected_samples)

    def create_test_dataset(self, class_encoding):
        indices_per_deployment = self.convert_challenge_annotations_to_yolo(class_encoding=class_encoding)
        selected_samples = self.select_all_background_labels(indices_per_deployment)
        self.create_spectrograms(selected_samples=selected_samples)

    def select_all_background_labels(self, indices_per_deployment):
        for dataset, indices_dataset in indices_per_deployment.items():
            background_indices = indices_dataset['background']
            indices_per_deployment[dataset]['selected_background'] = background_indices

            for selection in background_indices:
                shutil.move(self.labels_folder.joinpath('backgrounds', selection + '.txt'),
                            self.labels_folder.joinpath(selection + '.txt'))

        return indices_per_deployment

    def select_background_labels(self, indices_per_deployment):
        for dataset, indices_dataset in indices_per_deployment.items():
            label_indices = indices_dataset['labels']
            background_indices = indices_dataset['background']
            n_labels = min(len(label_indices), len(background_indices))
            print(f'There are {n_labels} labels in the dataset {dataset}')

            selected_background = random.sample(background_indices, n_labels)
            indices_per_deployment[dataset]['selected_background'] = selected_background

            for selection in selected_background:
                shutil.move(self.labels_folder.joinpath('backgrounds', selection + '.txt'),
                            self.labels_folder.joinpath(selection + '.txt'))

        return indices_per_deployment

    def create_spectrograms(self, selected_samples, overwrite=True):
        # First, create all the images
        print('Creating spectrograms...')
        for dataset, indices_dataset in selected_samples.items():
            selected_indices = indices_dataset['selected_background'] + indices_dataset['labels']
            for sample_i in tqdm(selected_indices):
                img_path = self.images_folder.joinpath(sample_i + '.png')
                wav_name = '_'.join(sample_i.split('_')[1:3])
                wav_path = self.wavs_folder.joinpath(dataset, wav_name + '.wav')
                i = float(sample_i.split('_')[-1])
                if overwrite or (not img_path.exists()):
                    start_chunk = int(i * self.blocksize)
                    chunk, fs = torchaudio.load(wav_path, normalize=True, frame_offset=start_chunk,
                                                num_frames=self.blocksize)
                    chunk = chunk[0, :]

                    if len(chunk) < self.blocksize:
                        chunk = F_general.pad(chunk, (0, self.blocksize - len(chunk)))
                    img, f = self.create_chunk_spectrogram(chunk)

                    if self.log:
                        fig, ax = plt.subplots()
                        ax.pcolormesh(img[:, :, ::-1])
                        ax.set_yscale('symlog')
                        plt.axis('off')
                        plt.ylim(bottom=3)
                        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                    else:
                        Image.fromarray(np.flipud(img)).save(img_path)
                    plt.close()
                i += self.overlap

    def create_chunk_spectrogram(self, chunk):
        sos = scipy.signal.iirfilter(20, [5, 124], rp=None, rs=None, btype='band',
                                     analog=False, ftype='butter', output='sos',
                                     fs=self.desired_fs)
        chunk = scipy.signal.sosfilt(sos, chunk)
        f, t, sxx = scipy.signal.spectrogram(chunk, fs=self.desired_fs, window=('hann'),
                                             nperseg=self.win_len,
                                             noverlap=self.win_overlap, nfft=self.nfft,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        sxx = 1 - sxx
        per = np.percentile(sxx.flatten(), 98)
        sxx = (sxx - sxx.min()) / (per - sxx.min())
        sxx[sxx > 1] = 1
        img = np.array(sxx * 255, dtype=np.uint8)
        return img, f

    def convert_challenge_annotations_to_yolo(self, class_encoding=None):
        """
        :param annotations_file:
        :param labels_to_exclude: list
        :param class_encoding: should be a dict with the name of the Tag as a key and an int as the value, for the
        yolo classes
        :return:
        """
        f_bandwidth = (self.desired_fs / 2) - self.F_MIN
        indices_per_deployment = {}
        for dataset_raw_folder in list(self.wavs_folder.glob('*')):
            if dataset_raw_folder.is_dir():
                dataset_name = dataset_raw_folder.name
                background_indices = []
                labels_indices = []

                selections_path = self.annotations_folder.joinpath(dataset_name + '.csv')
                selections = pd.read_csv(selections_path, parse_dates=['start_datetime', 'end_datetime'])

                # Get rid of all annotations which are not in the class_encoding list 
                selections = selections.loc[selections.annotation.isin(class_encoding.keys())]

                selections.loc[selections['low_frequency'] < self.F_MIN, 'low_frequency'] = self.F_MIN
                selections['height'] = (selections['high_frequency'] - selections['low_frequency']) / f_bandwidth

                # The y is from the TOP!
                selections['y'] = 1 - (selections['high_frequency'] / f_bandwidth)

                # Deal with datetime
                selections['start_datetime_wav'] = pd.to_datetime(selections['filename'].apply(lambda y: y.split('.')[0]),
                                                                    format='%Y-%m-%dT%H-%M-%S_%f')
                selections['start_datetime_wav'] = selections['start_datetime_wav'].dt.tz_localize('UTC')
                selections['start_datetime'] = selections['start_datetime'].dt.tz_localize('UTC')
                selections['end_datetime'] = selections['end_datetime'].dt.tz_localize('UTC')

                selections['start_seconds'] = (selections.start_datetime - selections.start_datetime_wav).dt.total_seconds()
                selections['end_seconds'] = (selections.end_datetime - selections.start_datetime_wav).dt.total_seconds()

                # save audio snippet information for wav files with annotations
                pbar = tqdm(total=len(list(self.wavs_folder.joinpath(dataset_name).glob('**/*.wav'))))

                for wav_file_path in list(self.wavs_folder.joinpath(dataset_name).glob('**/*.wav')):
                    wav_name = wav_file_path.name
                    waveform_info = torchaudio.info(wav_file_path)

                    wav_selections = selections.loc[selections.filename == wav_name]

                    i = 0.0
                    while (i * self.duration) < (waveform_info.num_frames / waveform_info.sample_rate):
                        start_seconds = i * self.duration
                        end_seconds = start_seconds + self.duration

                        start_mask = (wav_selections['start_seconds'] >= start_seconds) & (wav_selections[
                                                                                                'start_seconds'] <= end_seconds)
                        end_mask = (wav_selections['start_seconds'] >= start_seconds) & (wav_selections[
                                                                                                'end_seconds'] <= end_seconds)
                        chunk_selection = wav_selections.loc[start_mask | end_mask]
                        chunk_selection = chunk_selection.assign(start_x=((chunk_selection['start_seconds'] - i * self.duration) / self.duration).clip(lower=0, upper=1).values)
                        chunk_selection = chunk_selection.assign(end_x=((chunk_selection['end_seconds'] - i * self.duration) / self.duration).clip(lower=0, upper=1).values)

                        # compute the width in pixels
                        chunk_selection = chunk_selection.assign(width=(chunk_selection['end_x'] - chunk_selection['start_x']).values)

                        # Save the chunk detections so that they are with the yolo format
                        # <class > < x > < y > < width > < height >
                        chunk_selection = chunk_selection.assign(x=(chunk_selection['start_x'] + chunk_selection['width'] / 2).values)
                        chunk_selection.loc[:, 'y'] = (chunk_selection['y'] + chunk_selection['height'] / 2).values

                        # if ((chunk_selection.x + chunk_selection.width/2) > 1).sum() > 0 or (chunk_selection.y > 1).sum() > 0:
                        #     print(chunk_selection)
                        #     print(start_seconds, end_seconds)
                        chunk_selection = chunk_selection.replace(to_replace=class_encoding).infer_objects(copy=False)
                        new_name = dataset_name + '_' + wav_name.replace('.wav', '_%s' % i)

                        if len(chunk_selection) > 0:
                            labels_indices.append(new_name)
                            label_path = self.labels_folder.joinpath(new_name + '.txt')
                        else:
                            background_indices.append(new_name)
                            label_path = self.labels_folder.joinpath('backgrounds', new_name + '.txt')

                        chunk_selection[[
                            'annotation',
                            'x',
                            'y',
                            'width',
                            'height']].to_csv(label_path, header=None, index=None, sep=' ', mode='w')
                        # Add the station if the image adds it as well!
                        i += self.overlap
                    pbar.update(1)

                
                    indices_per_deployment[dataset_name] = {'background': background_indices, 'labels': labels_indices}
                pbar.close()
     
        return indices_per_deployment

    def convert_yolo_detections_to_csv(self, predictions_folder, reverse_class_encoding):
        # Convert to DataFrame
        labels_folder = predictions_folder.joinpath('labels')

        columns = ['dataset', 'filename', 'annotation', 'low_frequency', 'high_frequency',
                   'start_datetime', 'end_datetime', 'confidence', 'offset_i']

        detections_list = []
        f_bandwidth = (self.desired_fs / 2) - self.F_MIN
        for txt_label in tqdm(labels_folder.glob('*.txt'), total=len(list(labels_folder.glob('*.txt')))):
            name_parts = txt_label.name.split('_')
            wav_name = '_'.join(name_parts[1:-1]) + '.wav'
            dataset_name = name_parts[0]

            offset_i = float(name_parts[-1].split('.txt')[0])
            detections_i = pd.read_table(txt_label, header=None, sep=' ', names=['annotation', 'x', 'y',
                                                                                 'width', 'height', 'confidence'])
            detections_i['filename'] = wav_name
            detections_i['dataset'] = dataset_name
            detections_i['offset_i'] = offset_i
            detections_list.append(detections_i)

        detections = pd.concat(detections_list, ignore_index=True)

        # start and end in seconds from beginning of file
        detections['start_seconds'] = (detections.x - detections.width / 2 + detections.offset_i) * self.duration
        detections['end_seconds'] = detections.width * self.duration + detections['start_seconds']

        # start of wav file datetime
        detections['start_datetime_wav'] = pd.to_datetime(detections['filename'].apply(lambda y: y.split('.')[0]),
                                                          format='%Y-%m-%dT%H-%M-%S_%f')
        detections['start_datetime_wav'] = detections['start_datetime_wav'].dt.tz_localize('UTC')

        # Compute absolute start and end time
        detections['start_datetime'] = detections['start_datetime_wav'] + pd.to_timedelta(detections['start_seconds'],
                                                                                          unit='s')
        detections['end_datetime'] = detections['start_datetime_wav'] + pd.to_timedelta(detections['end_seconds'],
                                                                                        unit='s')

        detections['start_datetime'] = detections['start_datetime'].apply(lambda x: x.isoformat(timespec='microseconds'))
        detections['end_datetime'] = detections['end_datetime'].apply(lambda x: x.isoformat(timespec='microseconds'))

        # Frequency boundaries
        detections['low_frequency'] = ((1 - (detections.y + detections.height / 2)) * f_bandwidth).clip(lower=0)
        detections['high_frequency'] = ((1 - (detections.y - detections.height / 2)) * f_bandwidth).clip(upper=self.desired_fs/2)

        # Change the annotation names
        detections['annotation'] = detections['annotation'].replace(to_replace=reverse_class_encoding).infer_objects(copy=False)
        detections[columns].to_csv(self.path_to_dataset.joinpath('predictions.csv'), index=False)

        return detections


if __name__ == '__main__':
    path_to_dataset = input('Where is the dataset folder?')

    train_mode = input('Is it for the training dataset y/n?') == 'y'
    config_path = './baselines/yolo/dataset_config.json'
    #config_path = './dataset_config.json'

    f = open(config_path)
    config = json.load(f)

    ds = YOLODataset(config, path_to_dataset)
    if train_mode:
        ds.create_train_dataset(class_encoding=config['class_encoding'])
    else:
        ds.create_test_dataset(class_encoding=config['class_encoding'])

