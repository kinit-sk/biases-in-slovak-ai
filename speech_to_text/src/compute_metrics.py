import copy
import logging
import os
import sys
from typing import Optional

import pandas as pd

import click

import bisk_metrics
import config


@click.command()
@click.option('-p', '--predictions-path',
              help='file path containing predictions and pre-computed error rates',
              required=True)
@click.option('-o', '--output-dirpath',
              help='directory path to save the computed metrics to',
              required=True)
@click.option('-n', '--n-repetitions',
              help='number of times the predictions are resampled during bootstrapping',
              required=False,
              default=1000)
@click.option('-r', '--random-state',
              help='fixed random state to use for bootstrapping; omit if you do not want to use fixed random state',
              required=False,
              default=None)
@click.option('-j', '--n-jobs',
              help='number of parallel jobs to run when computing each metric individually',
              required=False,
              default=1)
@click.option('-s', '--save-indexes',
              help='if specified, save indexes of samples during bootstrapping for each repetition',
              is_flag=True,
              required=False,
              default=False)
@click.option('--include-test-set-only-scenarios/--do-not-include-test-set-only-scenarios',
              help='if specified, also compute metrics for the test set if a column indicating subsets exists in the data',
              required=False,
              default=True)
def main(
      predictions_path: str,
      output_dirpath: str,
      n_repetitions: int,
      random_state: Optional[int],
      n_jobs: int,
      save_indexes: bool,
      include_test_set_only_scenarios: bool,
):
  logger = logging.getLogger(__file__)
  logging.basicConfig(format='%(asctime)s %(message)s')
  logger.addHandler(logging.StreamHandler())
  logger.setLevel('INFO')

  df = pd.read_parquet(predictions_path)

  if config.GENDER_COL not in df.columns:
    logger.error(f'Column "{config.GENDER_COL}" not found in the dataset')
    sys.exit(1)

  _precompute_metrics_in_preparation_for_macro_average(df)

  male_and_female_cond = df[config.GENDER_COL].isin(['male', 'female'])
  male_cond = df[config.GENDER_COL] == 'male'
  female_cond = df[config.GENDER_COL] == 'female'
  recorded_by_male_and_female_cond = df[config.HAS_SENTENCE_MALE_AND_FEMALE_GENDER_COL]

  scenarios = [
    {
      'message': 'Computing metrics for all genders',
      'df': df,
      'suffix': '_all_genders',
    },
    {
      'message': 'Computing metrics for male and female gender',
      'df': df[male_and_female_cond],
      'suffix': '_male_and_female',
    },
    {
      'message': 'Computing metrics for male gender only',
      'df': df[male_cond],
      'suffix': '_male_only',
    },
    {
      'message': 'Computing metrics for female gender only',
      'df': df[female_cond],
      'suffix': '_female_only',
    },
    {
      'message': 'Computing metrics for transcriptions recorded by both male and female gender',
      'df': df[recorded_by_male_and_female_cond & male_and_female_cond],
      'suffix': '_recorded_by_male_and_female__male_and_female',
    },
    {
      'message': 'Computing metrics for transcriptions recorded by both male and female gender, male gender only',
      'df': df[recorded_by_male_and_female_cond & male_cond],
      'suffix': '_recorded_by_male_and_female__male_only',
    },
    {
      'message': 'Computing metrics for transcriptions recorded by both male and female gender, female gender only',
      'df': df[recorded_by_male_and_female_cond & female_cond],
      'suffix': '_recorded_by_male_and_female__female_only',
    },
  ]

  if len(df) == len(df[male_and_female_cond]):
    # We would obtain the same results
    del scenarios[0]

  if config.AGE_GROUP_COL in df.columns:
    new_scenarios = []

    for scenario in scenarios:
      age_groups_per_scenario = [item for item in df[config.AGE_GROUP_COL].unique() if not pd.isnull(item)]

      for age_group in age_groups_per_scenario:
        new_scenario = copy.copy(scenario)
        new_scenario['message'] += f', age group "{age_group}"'
        new_scenario['df'] = new_scenario['df'][new_scenario['df'][config.AGE_GROUP_COL] == age_group]
        new_scenario['suffix'] += f'__age_{age_group}'

        new_scenarios.append(new_scenario)

    scenarios.extend(new_scenarios)

  if include_test_set_only_scenarios and config.SUBSET_COL in df.columns:
    new_scenarios = []

    for scenario in scenarios:
      new_scenario = copy.copy(scenario)
      new_scenario['message'] += ', test set only'
      new_scenario['df'] = new_scenario['df'][new_scenario['df'][config.SUBSET_COL] == config.TEST_SET]
      new_scenario['suffix'] += '__test_set'

      new_scenarios.append(new_scenario)

    scenarios.extend(new_scenarios)

  os.makedirs(output_dirpath, exist_ok=True)

  metrics = {}
  
  for scenario in scenarios:
    message = scenario.pop('message', None)
    if message is not None:
      logger.info(message)

    metrics_for_scenario = bisk_metrics.compute_metrics_via_bootstrapping_for_all_predicted_columns(
      **scenario,
      n_repetitions=n_repetitions,
      random_state=random_state,
      n_jobs=n_jobs,
      save_indexes=save_indexes,
    )

    # Save intermediate results in case of a failure to avoid recomputing everything from scratch.
    pd.DataFrame(metrics_for_scenario).to_parquet(os.path.join(output_dirpath, f'metrics__{scenario["suffix"]}.parquet'))

    metrics.update(metrics_for_scenario)

  pd.DataFrame(metrics).to_parquet(os.path.join(output_dirpath, 'metrics.parquet'))

  logger.info('Done!')


def _precompute_metrics_in_preparation_for_macro_average(df):
  """Computes metrics for each pair of predicted and target transcription.

  These per-row metrics are used for computing macro-average when later performing bootstrapping.
  This saves computational resources by avoiding recomputing the metrics from the same pairs multiple times.
  """
  bisk_metrics.compute_metrics_per_row(df)


if __name__ == '__main__':
  main()
