import numpy as np
import pandas as pd
import os
import pathlib

joining_dict = {'bma': 'bma',
                'bmb': 'bmb',
                'bmz': 'bmz',
                'Bm-A': 'bma',
                'Bm-B': 'bmb',
                'Bm-Z': 'bmz',
                'bmd': 'fm',
                'bpd': 'fm',
                'Bp-40Down': 'fm',
                'Bm-D': 'fm',
                'bp20': 'bp20',
                'bp20plus': 'bp20plus',
                'Bp-20': 'bp20',
                'Bp-20Plus': 'bp20plus',
                'Not sure at all': 1,
                'Quite sure': 2,
                '100% sure!': 3}

def join_annotations_if_dir(path_to_annotations):
    """
    Join all the annotations of one directory, and return as pandas DataFrame

    :param path_to_annotations: pathlib.Path pointing to the folder with all the csv prediction files to evaluate
    or only one of them (either folder or file are good)
    :return: pandas.DataFrame with all the annotations combined
    """
    if path_to_annotations.is_dir():
        annotations_list = []
        for annotations_path in path_to_annotations.glob('*.csv'):
            annotations = pd.read_csv(annotations_path, parse_dates=['start_datetime', 'end_datetime'])
            annotations_list.append(annotations)
        total_annotations = pd.concat(annotations_list, ignore_index=True)
    else:
        total_annotations = pd.read_csv(path_to_annotations, parse_dates=['start_datetime', 'end_datetime'])

    return total_annotations


def run(annotations_path, iou_threshold=0.3):
    """
    Run the evaluation protocol. Prints the obtained results per dataset and in total

    :param annotations_path: str or pathlib.Path pointing to the folder with all the csv annotations files to evaluate
    or only one of them (either folder or file are good)

    :param iou_threshold: float, 0 to 1 for the IOU threshold to consider for evaluation
    :return: None
    """

    if type(annotations_path) is str:
        annotations_path = pathlib.Path(annotations_path)
    annotations = join_annotations_if_dir(annotations_path)

    # Read annotation status file
    annot_status = pd.read_csv(
        os.path.join(annotations_path.parents[0], annotations['dataset'].iloc[0] + '_status.csv'))
    annot_status = annot_status.replace('FINISHED', '0')
    annot_status = annot_status.replace('UNASSIGNED', 'NA')
    annot_status = annot_status.replace('CREATED', 'NA')
    annot_status['unknown'] = 0
    annotators = annot_status.columns[2:len(annot_status.columns)].tolist()
    annot_status.iloc[:, 2:len(annot_status.columns)] = annot_status.iloc[:, 2:len(annot_status.columns)].astype('object')


    #uniform labels, remove irrelevant rows, add columns for doublet annotations, sort annotations by datetime and reset index, add info on annotation status
    annotations = annotations.replace(joining_dict)
    annotations['filename'] = annotations['filename'].str.replace('T','_')
    annotations['filename'] = annotations['filename'].str.replace('-','_')
    annotations['filename'] = annotations['filename'].str.replace('_000','')
    annotations['dataset'] = annotations['dataset'].str.replace('k','K')
    annotations = annotations.fillna(value = {"confidence_indicator_label": 2})
    annotations = annotations.drop(annotations[annotations.type == 'WEAK'].index)
    annotations = annotations.drop(annotations[annotations.annotation == 'nan'].index)
    annotations = annotations.sort_values(by='start_datetime').reset_index(drop=True)
    annotations['n_matches'] = 0
    annotations['summed_confidence'] = 0
    annotations = annotations.merge(annot_status.loc[:,annot_status.columns[1:len(annot_status.columns)].tolist()], how='inner', on='filename',suffixes=(False, False),copy=False)
    annotations['relative_matches'] = 0
    annotations['relative_confidence'] = 0

    #Create empty dataframe for relevant annotations
    merged_annotations = pd.DataFrame(columns = annotations.columns)


    #compare row to all the other rows
    for i, row in annotations.iterrows():
        timefilt_annot = annotations.loc[(annotations['start_datetime'] <= (row.start_datetime + pd.Timedelta(minutes=1))) &
                                            (annotations['start_datetime'] >= (row.start_datetime - pd.Timedelta(minutes=1)))]
        timefilt_annot['iou'] = 0
        for j, other_row in timefilt_annot.iterrows():
             if i != j & j in timefilt_annot.index:
                # For each row, compute the minimum end and maximum start with all the ground truths
                min_end = np.minimum(row['end_datetime'], other_row['end_datetime'])
                max_start = np.maximum(row['start_datetime'], other_row['start_datetime'])
                inter = (min_end - max_start).total_seconds()
                union = (row['end_datetime'] - row['start_datetime']).total_seconds() + (
                    (other_row['end_datetime'] - other_row['start_datetime']).total_seconds()) - inter
                timefilt_annot.iou[j] = inter / union


        # if no iou matches just collect the row itself
        # if there is one or more IOU matches collect the row in other table and write number of matches plus summed level of confidence, relative matches (divided by N annotators) and relative confidence (divided by N annotators):
        # if label matches between all iou matches collect row with this label
        # if labels are mix of bma, bmb, and/or bmz collect row and replace label by bmabz
        # if labels are mix of bp20 and bp20plus collect row and replace label with bp20
        # else (if labels don't match at all) collect row itself
        if not any(timefilt_annot['iou'] >= iou_threshold) and i in timefilt_annot.index:
            row['n_matches'] = 1
            row['summed_confidence'] = row['confidence_indicator_label']
            row[row['annotator']] = 1
            row['relative_matches'] = row['n_matches'] / sum(
                ~np.isnan(pd.to_numeric(row[annotators], errors='coerce')))
            row['relative_confidence'] = row['summed_confidence'] / sum(
                ~np.isnan(pd.to_numeric(row[annotators], errors='coerce')))
            merged_annotations = merged_annotations._append(row.to_frame().T)
            annotations = annotations.drop(annotations[annotations['start_datetime'] == row.start_datetime].index)
        if any(timefilt_annot['iou'] >= iou_threshold) and i in timefilt_annot.index:
            temp_annot = row.to_frame().T
            temp_annot = temp_annot._append(timefilt_annot[(timefilt_annot['iou'] >= iou_threshold) &
                                                           (timefilt_annot['annotator']!= row.annotator)].drop('iou',axis=1))
            temp_annot['n_matches'].iloc[0] = temp_annot.shape[0]
            temp_annot['summed_confidence'].iloc[0] = sum(temp_annot['confidence_indicator_label'])
            temp_annot[temp_annot['annotator'].tolist()] = 1
            temp_annot['relative_matches'].iloc[0] = temp_annot['n_matches'].iloc[0] / sum(
                ~np.isnan(pd.to_numeric(temp_annot[annotators].iloc[0], errors='coerce')))
            temp_annot['relative_confidence'].iloc[0] = temp_annot['summed_confidence'].iloc[0] / sum(
                ~np.isnan(pd.to_numeric(temp_annot[annotators].iloc[0], errors='coerce')))
            if temp_annot['annotation'].nunique() == 1:
                merged_annotations = merged_annotations._append(temp_annot.iloc[0])
                annotations = annotations.drop(annotations[annotations['start_datetime'] == row.start_datetime].index)
                try:
                    annotations = annotations.drop(timefilt_annot[(timefilt_annot['iou'] >= iou_threshold) &
                                                           (timefilt_annot['annotator'] != row.annotator)].index,axis=0)
                except:
                    print('row already dropped becasue annotations had the exact same start_datetime')
            elif set(temp_annot['annotation']).issubset(['bma','bmb','bmz']):
                temp_annot['annotation'].iloc[0] = 'bmabz'
                merged_annotations = merged_annotations._append(temp_annot.iloc[0])
                annotations = annotations.drop(annotations[annotations['start_datetime'] == row.start_datetime].index)
                try:
                    annotations = annotations.drop(timefilt_annot[(timefilt_annot['iou'] >= iou_threshold) &
                                                                  (timefilt_annot['annotator'] != row.annotator)].index,axis=0)
                except:
                    print('row already dropped becasue annotations had the exact same start_datetime')
            elif set(temp_annot['annotation']).issubset(['bp20','bp20plus']):
                temp_annot['annotation'].iloc[0] = 'bp20'
                merged_annotations = merged_annotations._append(temp_annot.iloc[0])
                annotations = annotations.drop(annotations[annotations['start_datetime'] == row.start_datetime].index)
                try:
                    annotations = annotations.drop(timefilt_annot[(timefilt_annot['iou'] >= iou_threshold) &
                                                                  (timefilt_annot['annotator'] != row.annotator)].index,axis=0)
                except:
                    print('row already dropped becasue annotations had the exact same start_datetime')
            else:
                row['n_matches'] = 1
                row['summed_confidence'] = row['confidence_indicator_label']
                row[row['annotator']] = 1
                row['relative_matches'] = row['n_matches'] / sum(
                    ~np.isnan(pd.to_numeric(row[annotators], errors='coerce')))
                row['relative_confidence'] = row['summed_confidence'] / sum(
                    ~np.isnan(pd.to_numeric(row[annotators], errors='coerce')))
                merged_annotations = merged_annotations._append(row.to_frame().T)
                annotations = annotations.drop(annotations[annotations['start_datetime'] == row.start_datetime].index)

    merged_annotations.to_csv(os.path.join(annotations_path, merged_annotations['dataset'].iloc[0] + '_joined.csv'), index=False)
    return(merged_annotations)


if __name__ == '__main__':
    annotations_csv_path = pathlib.Path(input('Where are the annotations in csv format?'))
    run(annotations_csv_path)