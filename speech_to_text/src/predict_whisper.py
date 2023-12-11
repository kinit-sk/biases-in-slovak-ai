import os

os.environ['TRANSFORMERS_CACHE'] = '/data/.cache'

import click

import pandas as pd
import tqdm

import whisper

import bisk_predict
import config


@click.command()
@click.option('-a', '--audio-path', help='directory path containing audio samples', required=True)
@click.option('-s', '--sentences-path', help='path to a parquet file containing input sentences', required=True)
@click.option('-m', '--model-size', help='size of the Whisper model', required=True)
@click.option('-o', '--output-path', help='path to a parquet file containing input sentences plus predicted sentences', required=True)
def main(
      audio_path: str,
      sentences_path: str,
      model_size: str,
      output_path: str,
):
  tqdm.tqdm.pandas()

  print(f'Downloading Whisper model, size {model_size}')

  model = whisper.load_model(model_size)

  raw_predicted_col = f'{config.PREDICTED_RAW_COL_PREFIX}whisper-{model_size}_lang-sk'
  predicted_col = f'{config.PREDICTED_COL_PREFIX}whisper-{model_size}_lang-sk'

  df_with_predictions = pd.read_parquet(sentences_path)

  print(f'Predicting with Slovak language explicitly specified on input')

  bisk_predict.predict_whisper(
    df_with_predictions,
    model,
    config.AUDIO_PATH_COL,
    audio_path,
    'sk',
    raw_predicted_col,
    predicted_col,
  )

  raw_predicted_col = f'{config.PREDICTED_RAW_COL_PREFIX}whisper-{model_size}_lang-auto'
  predicted_col = f'{config.PREDICTED_COL_PREFIX}whisper-{model_size}_lang-auto'

  print(f'Predicting with automatic language recognition')

  bisk_predict.predict_whisper(
    df_with_predictions,
    model,
    config.AUDIO_PATH_COL,
    audio_path,
    None,
    raw_predicted_col,
    predicted_col,
  )

  df_with_predictions.to_parquet(output_path)

  print('Done!')


if __name__ == '__main__':
  main()
