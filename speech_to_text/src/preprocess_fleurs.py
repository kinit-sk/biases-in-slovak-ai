from typing import Optional

import os
import pathlib
import shutil

import pandas as pd

import click
import tqdm

import datasets

import bisk_audio
import bisk_preprocessing
import config


@click.command()
@click.option('-d', '--data-dirpath', help='directory path containing dataset', required=True)
@click.option('-o', '--output-path', help='path containing preprocessed dataset', required=True)
def main(
      data_dirpath: str,
      output_path: str,
):
  tqdm.tqdm.pandas()

  data_dirpath = os.path.normpath(data_dirpath)

  print('Downloading dataset')

  fleurs_asr_sk = datasets.load_dataset('google/xtreme_s', 'fleurs.sk_sk', cache_dir=data_dirpath)

  print('Preprocessing dataset')

  df = _merge_subsets(fleurs_asr_sk)

  df = _add_gender_col(df)

  df = df.rename(columns={'path': config.AUDIO_PATH_COL})

  df[config.AUDIO_PATH_COL] = df[config.AUDIO_PATH_COL].apply(lambda path: os.path.relpath(path, data_dirpath))

  bisk_preprocessing.preprocess_transcriptions(df, 'transcription', config.GROUND_TRUTH_COL)

  df = bisk_preprocessing.add_has_sentence_male_and_female_gender(df, config.GROUND_TRUTH_COL, config.GENDER_COL)

  print('Obtaining audio metadata')

  df = df.apply(bisk_audio.get_audio_metadata, axis=1, args=(data_dirpath, config.AUDIO_PATH_COL))

  print(f'Saving preprocessed file to "{output_path}"')

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  df.to_parquet(output_path)


def _merge_subsets(fleurs_asr):
  df_train = fleurs_asr['train'].to_pandas()
  df_train['subset'] = 'train'

  df_validation = fleurs_asr['validation'].to_pandas()
  df_validation['subset'] = 'validation'

  df_test = fleurs_asr['test'].to_pandas()
  df_test['subset'] = 'test'

  df = pd.concat([df_train, df_validation, df_test], ignore_index=True)

  return df


def _add_gender_col(df):
  df = df.rename(columns={config.GENDER_COL: 'gender_number'})

  df[config.GENDER_COL] = 'unknown'
  df.loc[df['gender_number'] == 0, config.GENDER_COL] = 'male'
  df.loc[df['gender_number'] == 1, config.GENDER_COL] = 'female'

  return df


if __name__ == '__main__':
  main()
