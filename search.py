"""
This script is just for searching for variables in the dataset. 
"""
from rpy2.robjects import pandas2ri
import rpy2.robjects as robj
from scipy import stats
import pandas as pd
import numpy as np
import os.path
import os
import re


def get_variable_dictionaries():
    """
    1) Constructs a dataframe of variable names and definitions from catalogue csv data. 
    2) Creates summary dataframes for questionnaire metadata (for catalogue and for 
        available features). 
    3) Keeps track of variables in available data that are missing in the catalogue. 
    """

    # file/folder paths 
    catalogue_path = '~/alspac/workspace/variable_catalogue/csv'
    thresholds_path = '~/alspac/workspace/thresholded'
    feat_dir = '~/alspac/workspace/features'
    feat_counts_path = '~/alspac/workspace/na_counts.csv'
    catalogue_path, thresholds_path, feat_dir, feat_counts_path = map(os.path.expanduser,
                                                                      [catalogue_path, thresholds_path, feat_dir,
                                                                       feat_counts_path])

    # xlsx and csv paths for summaries and catalogues
    diff_name, full_name, cat_name = 'diffed_summary', 'full_summary', 'catalogue'
    diff_path_x, full_path_x, cat_path_x = map(lambda u: feat_dir + os.sep + u + '.xlsx',
                                               [diff_name, full_name, cat_name])
    diff_path, full_path, cat_path = map(lambda u: feat_dir + os.sep + 'csv' + os.sep + u + '_',
                                         [diff_name, full_name, cat_name])

    ######################### CATALOGUE #########################

    # construct dictionary from variable catalogue
    frames = list()
    for subdir, dirs, files in os.walk(catalogue_path):
        for file in files:
            file_path = subdir + os.sep + file

            # get detailed description from first line
            with open(file_path, newline='') as f:
                top = [k.strip('\"') for k in f.readline().rstrip().split(',') if k]
                assert len(top) > 0
                desc = '. '.join(top)

            # read to dataframe
            df = pd.read_csv(file_path, header=2)
            df = df.loc[:, df.columns[0:3]]
            df.columns = ['file_name', 'name', 'details']
            df.dropna(axis=0, how='all', inplace=True)

            # add columns with metadata
            df['type'] = file.split(os.extsep)[0]
            df['file_descriptor'] = top[0]
            df['context'] = desc

            frames.append(df)

            # stack dataframes
    var_df = pd.concat(frames, ignore_index=True)

    # append ID variables 
    var_df = var_df.append([{
        'file_name': 'GLOBAL',
        'name': 'qlet',
        'details': 'Identifies children from multiple births.',
        'type': 'global',
        'file_descriptor': 'GLOBAL',
        'context': 'Identifies children from multiple births.',
    },
        {
            'file_name': 'GLOBAL',
            'name': 'cidB2855',
            'details': 'Identifier for each child. Combine with qlet to completely identify a child.',
            'type': 'global',
            'file_descriptor': 'GLOBAL',
            'context': 'Identifier for each child. Combine with qlet to completely identify a child.',
        }
    ], ignore_index=True)

    # append DAWBA band variables
    bands = ['levelband_15', 'pextband_15', 'padhdbandd_15', 'padhdbandi_15', 'pbehavband_15',
             'poddband_15', 'pcdband_15', 'semotband_15', 'sdepband_15', 'sanxband_15', 'sgenaband_15',
             'spanband_15', 'sagoband_15', 'sptsdband_15', 'ssophband_15', 'sspphband_15', 'any01_15',
             'pext01_15', 'padhd01_15', 'phk01_15', 'pbehav01_15', 'podd01_15', 'pcd01_15', 'semot01_15',
             'sdep01_15', 'sanx01_15', 'sgena01_15', 'span01_15', 'sago01_15', 'sptsd01_15', 'ssoph01_15',
             'sspph01_15']
    for b in bands:
        var_df = var_df.append([{
            'file_name': 'DAWBA Bands',
            'name': b,
            'details': 'Based on the DAWBA questions, ALSPAC derived placements for each individual in one of five risk bands for each disorder.',
            'type': 'uncatalogued',
            'file_descriptor': 'DAWBA Bands',
            'context': 'DAWBA Bands',
        }], ignore_index=True)

    # get list of available features as data frame (includes missing counts) 
    fts = pd.read_csv(feat_counts_path, index_col=0)
    fts.reset_index(level=0, inplace=True)
    fts.rename(index=str, columns={'index': 'name', 'na_count': 'missing'}, inplace=True)
    fts['proportion'] = fts['missing'] / fts['total']

    # sort variables by questionnaire 
    var_df.sort_values(by=['type', 'file_descriptor', 'context'],
                       axis=0, ascending=True, inplace=True)

    # add missing counts and total count for available features
    var_df = pd.merge(var_df, fts, how='left', left_on=['name'], right_on=['name'])

    ######################### SUMMARIES #########################

    # questionnaire summarization  
    def summarize(df, complete=False):
        if complete:
            # don't compute statistics for complete catalogue 
            df = df.drop(['name', 'details', 'missing', 'total', 'proportion'], axis=1, inplace=False)
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.sort_values(by=['type', 'file_descriptor', 'context'],
                           axis=0, ascending=True, inplace=True)

            return df

        else:
            # remove total counts (only keep proportions)
            df = df.drop(['missing', 'total'], axis=1, inplace=False)

            # avoid harmonic mean errors
            df.proportion = df.proportion + 1e-6

            # group together by questionnaire file
            df = df.groupby(['type', 'file_descriptor', 'context']).agg({
                'proportion': ['count', stats.hmean, 'mean', 'min', 'median', 'max']
            })
            df.reset_index(inplace=True)

            # round statistics to 3 decimals
            df.proportion = df.proportion.round(decimals=3)

            # update column index 
            tstrip = lambda tup: tuple([x for x in tup if x])
            df.columns = ['_'.join(tstrip(tup)).strip() for tup in df.columns.values]

            # represent total count as int
            df.proportion_count = df.proportion_count.astype(int)

            # sort by questionnaire 
            df.sort_values(by=['type', 'file_descriptor', 'context'],
                           axis=0, ascending=True, inplace=True)

            return df

    # summary for entire catalogue
    sm_full = summarize(var_df, complete=True)

    # create excel file writers for summaries 
    diff_writer, full_writer, cat_writer = map(pd.ExcelWriter, [diff_path_x, full_path_x, cat_path_x])

    # save feature catalogue and catalogue summary (xlsx and csv)
    var_df.to_excel(cat_writer, 'Complete', index=False)
    var_df.to_csv(cat_path + 'Complete' + '.csv', index=False)
    sm_full.to_excel(full_writer, 'Complete', index=False)
    sm_full.to_csv(full_path + 'Complete' + '.csv', index=False)

    # get list of feature files
    files = [f for _, _, files in os.walk(thresholds_path) for f in files]
    sort_fn = lambda x: int(re.search(r'\d+', x).group())  # comparator 
    files.sort(key=sort_fn, reverse=True)  # sort file list 

    # creates new dataframe with rows in df2 that aren't in df1
    def diff(df1, df2):
        s_cols = df2.columns.tolist()  # summary columns 
        df1 = set([tuple(row) for row in df1.values])
        df2 = set([tuple(row) for row in df2.values])
        return pd.DataFrame(list(df2.difference(df1)), columns=s_cols)

    missing = list()
    summaries = list()

    # generate (diff'ed and full) summaries for each list of features 
    for i, file in enumerate(files):
        file_path = thresholds_path + os.sep + file

        # get feature list for this 
        feats = pd.read_csv(file_path, header=None, names=['feats']).feats

        # features missing from catalogue
        ms = feats[~feats.isin(var_df['name'])]
        missing.append(ms.tolist())

        # get subset of catalogue in feats 
        subset = var_df[var_df['name'].isin(feats)]

        # save catalogue subset
        subset.to_excel(cat_writer, file, index=False)
        subset.to_csv(cat_path + file + '.csv', index=False)

        # create questionnaire summaries for available features
        sm = summarize(subset)
        summaries.append(sm)

        # take diffs of feature summary sequence (avoids repetition)
        if i > 0:
            sm_prev = summaries[i - 1]
            sm = diff(sm_prev, sm)

            # save diff'ed summary
        sm.to_excel(diff_writer, file, index=False)
        sm.to_csv(diff_path + file + '.csv', index=False)

        # save full summary 
        summaries[-1].to_excel(full_writer, file, index=False)
        summaries[-1].to_csv(full_path + file + '.csv', index=False)

        # save workbooks
    diff_writer.save()
    full_writer.save()
    cat_writer.save()

    ######################### MISSING IN CATALOGUE #########################

    # aggregate uncatalogued features 
    missing = list({x for fts in missing for x in fts})
    missing.sort()

    # save uncatalogued
    pd.Series(missing).to_csv(feat_dir + os.sep + 'uncatalogued.csv', index=False)

    return var_df, sm_full, summaries, missing


if __name__ == '__main__':

    df, sm_full, summaries, missing = get_variable_dictionaries()

    # this function gets the dataframe row for the provided variable name:
    var = lambda x: df.loc[df.name == x]
