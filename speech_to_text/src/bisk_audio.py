import os

import torchaudio
import whisper


def get_audio_metadata(row, root_dirpath, audio_path_col):
  loaded_metadata = torchaudio.info(os.path.join(root_dirpath, row[audio_path_col]))

  row['sample_rate'] = loaded_metadata.sample_rate
  row['bits_per_sample'] = loaded_metadata.bits_per_sample
  row['num_channels'] = loaded_metadata.num_channels
  row['num_frames'] = loaded_metadata.num_frames
  row['encoding'] = loaded_metadata.encoding

  return row


def transcribe_audio_whisper(audio_rel_filepath, root_dirpath, model, language='sk'):
  return whisper.transcribe(model, os.path.join(root_dirpath, audio_rel_filepath), language=language)


def resample_audio(row, input_root_dirpath, audio_path_col, current_sampling_rate_col, output_root_dirpath, target_sampling_rate, **resample_kwargs):
  try:
    audio = torchaudio.load(os.path.join(input_root_dirpath, row[audio_path_col]))[0]
  except RuntimeError:
    row['audio_resample_success'] = False
  else:
    resampled_audio = torchaudio.functional.resample(audio, row[current_sampling_rate_col], target_sampling_rate, **resample_kwargs)

    torchaudio.save(os.path.join(output_root_dirpath, row[audio_path_col]), resampled_audio, target_sampling_rate)

    row['audio_resample_success'] = True

  return row
