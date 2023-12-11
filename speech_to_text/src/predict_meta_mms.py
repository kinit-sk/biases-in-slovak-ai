import os

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache'

import click

import pandas as pd
import tqdm

import bisk_predict
import config


@click.command()
@click.option('-a', '--audio-path', help='directory path containing audio samples', required=True)
@click.option('-s', '--sentences-path', help='path to a parquet file containing input sentences', required=True)
@click.option('-o', '--output-path', help='path to a parquet file containing input sentences plus predicted sentences', required=True)
def main(
      audio_path: str,
      sentences_path: str,
      output_path: str,
):
  tqdm.tqdm.pandas()

  predicted_col = f'{config.PREDICTED_COL_PREFIX}meta-mms_lang-sk'

  df_with_predictions = pd.read_parquet(sentences_path)

  print(f'Predicting with Slovak language explicitly specified on input')

  bisk_predict.predict_meta_mms(
    df_with_predictions,
    config.AUDIO_PATH_COL,
    audio_path,
    'slk',
    predicted_col,
  )

  predicted_col = f'{config.PREDICTED_COL_PREFIX}meta-mms_lang-auto'

  print(f'Predicting with automatic language recognition')

  bisk_predict.predict_meta_mms(
    df_with_predictions,
    config.AUDIO_PATH_COL,
    audio_path,
    None,
    predicted_col,
  )

  df_with_predictions.to_parquet(output_path)

  print('Done!')


if __name__ == '__main__':
  main()
