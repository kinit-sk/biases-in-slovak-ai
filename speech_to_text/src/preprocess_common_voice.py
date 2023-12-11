import os

import pandas as pd

import click
import tqdm

import bisk_audio
import bisk_preprocessing
import config


AUDIO_ORIGINAL_DIRNAME = 'clips'


@click.command()
@click.option('-d', '--data-dirpath', help='directory path containing dataset', required=True)
@click.option('-o', '--output-path', help='path containing preprocessed dataset', required=True)
def main(
      data_dirpath: str,
      output_path: str,
):
  tqdm.tqdm.pandas()

  print('Preprocessing dataset')

  os.makedirs(os.path.join(data_dirpath, config.AUDIO_DIRNAME), exist_ok=True)

  data_subdirpath = _find_data_subdirectory(data_dirpath)
  if data_subdirpath is None:
    data_subdirpath = data_dirpath

  df = pd.read_csv(os.path.join(data_subdirpath, 'validated.tsv'), sep='\t')

  df.loc[df[config.GENDER_COL].isnull(), config.GENDER_COL] = 'unknown'

  df = df.rename(columns={'sentence': config.GROUND_TRUTH_RAW_COL, 'path': config.AUDIO_PATH_COL})

  bisk_preprocessing.preprocess_transcriptions(df, config.GROUND_TRUTH_RAW_COL, config.GROUND_TRUTH_COL)

  df = bisk_preprocessing.add_has_sentence_male_and_female_gender(df, config.GROUND_TRUTH_COL, config.GENDER_COL)

  print('Obtaining original audio metadata')

  df = df.progress_apply(
    bisk_audio.get_audio_metadata,
    axis=1,
    args=(os.path.join(data_subdirpath, AUDIO_ORIGINAL_DIRNAME), config.AUDIO_PATH_COL))

  print(f'Converting audio samples to {config.PROCESSED_SAMPLING_RATE} sampling rate')

  df = df.progress_apply(
    bisk_audio.resample_audio,
    axis=1,
    args=(
      os.path.join(data_subdirpath, AUDIO_ORIGINAL_DIRNAME),
      config.AUDIO_PATH_COL,
      'sample_rate',
      os.path.join(data_dirpath, config.AUDIO_DIRNAME),
      config.PROCESSED_SAMPLING_RATE))

  df[config.AUDIO_PATH_COL] = df[config.AUDIO_PATH_COL].apply(lambda p: os.path.join(config.AUDIO_DIRNAME, p))

  print(f'Saving preprocessed file to "{output_path}"')

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  df.to_parquet(output_path)


def _find_data_subdirectory(data_dirpath):
  for subdirpath, _dirnames, filenames in os.walk(data_dirpath):
    for filename in filenames:
      if filename == 'validated.tsv':
        return subdirpath
  
  return None


if __name__ == '__main__':
  main()
