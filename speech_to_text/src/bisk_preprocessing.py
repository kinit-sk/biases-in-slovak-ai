import config


def preprocess_transcriptions(df, source_col, dest_col):
  df[dest_col] = df[source_col].str.lower()
  df[dest_col] = remove_unwanted_characters(df[dest_col])
  df[dest_col] = remove_punctuation_including_periods(df[dest_col])
  df[dest_col] = remove_extra_space(df[dest_col])
  df[dest_col] = df[dest_col].str.strip()


def remove_unwanted_characters(sentences):
  return sentences.str.replace(r'["“”„‟‘’‚‛\';]', ' ', regex=True)


def remove_punctuation(sentences):
  return sentences.str.replace(r'[,;:?!"\'`]', ' ', regex=True)


def remove_punctuation_including_periods(sentences):
  return sentences.str.replace(r'[,;:?!"\'`.]', ' ', regex=True)


def remove_extra_space(sentences):
  return sentences.str.replace(r'\s+', ' ', regex=True)


def add_has_sentence_male_and_female_gender(df, transcription_col, gender_col):
  def has_sentence_male_and_female_gender(rows):
    return len(rows['gender'].value_counts()) > 1

  new_col_name = config.HAS_SENTENCE_MALE_AND_FEMALE_GENDER_COL

  df_processed = df[df[gender_col].isin(['female', 'male'])]
  df_processed = df_processed[df_processed[transcription_col].duplicated(keep=False)]

  df_has_sentence_male_and_female_gender = df_processed.groupby(
    transcription_col, sort=False).apply(has_sentence_male_and_female_gender)
  df_has_sentence_male_and_female_gender.name = new_col_name
  df_has_sentence_male_and_female_gender = df_has_sentence_male_and_female_gender.reset_index()

  df_merged = df.merge(df_has_sentence_male_and_female_gender, on=transcription_col, how='left')

  df_merged[new_col_name] = df_merged[new_col_name].fillna(False)

  return df_merged


def get_word_count(df, transcription_col):
   return df[transcription_col].str.split(' ').apply(len).value_counts().sort_index(axis=0)
