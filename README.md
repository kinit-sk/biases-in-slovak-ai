# Data

The repository contains two main data files: (1) `annotations.csv` are the raw data as we filtered it and processed it, (2) `samples.txt` is the filtered final selection used in the experiments.

## `annotations.csv`

- `sentence` - The sentence proposed by the data creator.
- `stereotype` - The stereotype assigned to the `sentence`.
- `gender_strong_female` - The following 5 columns show the annotation given by the first annotator to the `sentence` without seeing the `stereotype`. The question was: Stereotypically, is the sentence female or male?
- `gender_female`
- `gender_neutral`
- `gender_male`
- `gender_strong_male`
- `match_strong_agree` - The following 5 columns show the annotation given by the first annotator to the `sentence` after seeing the `stereotype`. The question was: Does the stereotype match the sentence?
- `match_agree`
- `match_neutral`
- `match_disagree`
- `match_strong_disagree`
- `final_verdict` - The final decision given by the second annotator: _yes_, _fix_ or _no_.
- `final_stereotype` - The final stereotype id assigned by the second annotator.
- `comment` - A comment written by the first annotator.
- `fix` - A fixed sentence proposed by either annotator.
- `first_annotator` - Initials of the first data annotator.
- `second_annotator` - Initials of the second data annotator.
- `creator` - Initials of the data creator.

## `samples.txt`

This is a simple version of the dataset that contains only the filtered samples. Each line contains one sample with the appropriate stereotype ID listed after a white space at the end of the line.

# Running the code

## `Dockerfile`

...

## Translators

The code work with various paid machine translation services. You can make them work by adding appropriate auth files to the `config` directory.

- `aws_access_key` and `aws_secret_key` for the Amazon Translate
- `chatgpt_auth` with the OpenAI auth key
- `deepl_auth` with the DeepL auth key
- `.json` file with the Google Cloud Platform service account key

