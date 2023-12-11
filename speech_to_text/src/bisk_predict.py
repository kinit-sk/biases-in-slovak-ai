import os

import tqdm

import torch
import torchaudio
import transformers

import bisk_audio
import bisk_preprocessing


def predict_whisper(df, model, input_col, root_dirpath, language, raw_predicted_col, predicted_col):
  df[raw_predicted_col] = df[input_col].progress_apply(
    bisk_audio.transcribe_audio_whisper, args=(root_dirpath, model), language=language)

  df[predicted_col] = df[raw_predicted_col].apply(lambda x: x['text'])

  bisk_preprocessing.preprocess_transcriptions(df, predicted_col, predicted_col)


def predict_meta_mms(df, input_col, root_dirpath, language, predicted_col):
  model_id = 'facebook/mms-1b-fl102'

  processor = transformers.AutoProcessor.from_pretrained(model_id)
  model = transformers.Wav2Vec2ForCTC.from_pretrained(model_id)

  if language is not None:
    processor.tokenizer.set_target_lang(language)
    model.load_adapter(language)

  # Send model to GPU
  model.to('cuda:0')

  transcriptions = []

  for rel_path in tqdm.tqdm(df[input_col]):
    audio_data = torchaudio.load(os.path.join(root_dirpath, rel_path))[0][0]

    inputs = processor(audio_data, sampling_rate=16_000, return_tensors='pt')
    # Send data to GPU
    inputs.to('cuda:0')

    with torch.no_grad():
      outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcriptions.append(processor.decode(ids))

  df[predicted_col] = transcriptions

  bisk_preprocessing.preprocess_transcriptions(df, predicted_col, predicted_col)
