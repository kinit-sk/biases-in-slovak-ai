import collections

import joblib
import numpy as np
import scipy.stats

import torchmetrics

import tqdm

import config


def compute_metrics_per_row(df):
  tqdm.tqdm.pandas()

  wer = torchmetrics.text.WordErrorRate()
  cer = torchmetrics.text.CharErrorRate()

  cols = df.columns[
    df.columns.str.startswith(config.PREDICTED_COL_PREFIX)
    & ~df.columns.str.startswith(config.PREDICTED_RAW_COL_PREFIX)]

  for col in cols:
    wer_col_name = config.WER_PREFIX + col[len(config.PREDICTED_COL_PREFIX):]
    df[wer_col_name] = df.progress_apply(_compute_metric, args=(wer, col), axis=1)

    cer_col_name = config.CER_PREFIX + col[len(config.PREDICTED_COL_PREFIX):]
    df[cer_col_name] = df.progress_apply(_compute_metric, args=(cer, col), axis=1)


def _compute_metric(row, torch_metrics_obj, predicted_col):
  return torch_metrics_obj(row[predicted_col], row[config.GROUND_TRUTH_COL]).item()


def compute_metrics_via_bootstrapping_for_all_predicted_columns(
      df, suffix='', n_repetitions=1000, random_state=None, n_jobs=1, save_indexes=False):
  predicted_cols = [
    col for col in df.columns
    if col.startswith(config.PREDICTED_COL_PREFIX) and not col.startswith(config.PREDICTED_RAW_COL_PREFIX)]

  metrics_list = joblib.Parallel(n_jobs=n_jobs)(
    joblib.delayed(compute_metrics_via_bootstrapping)(
      df, col, suffix, n_repetitions, random_state, save_indexes)
    for col in predicted_cols
  )

  metrics = {}
  for item in metrics_list:
    metrics.update(item)

  return metrics


def compute_metrics_via_bootstrapping(
        df, predicted_col, suffix='', n_repetitions=1000, random_state=None, save_indexes=False):
  random_generator = np.random.default_rng(random_state)

  predicted_col_root = predicted_col[len(config.PREDICTED_COL_PREFIX):]

  wer_obj = torchmetrics.text.WordErrorRate()
  wer_col = config.WER_PREFIX + predicted_col_root

  cer_obj = torchmetrics.text.CharErrorRate()
  cer_col = config.CER_PREFIX + predicted_col_root

  metrics = collections.defaultdict(list)

  for _ in range(0, n_repetitions):
    sampled_indexes = random_generator.choice(df.index, size=len(df.index))

    predicted_transcriptions_sampled = df.loc[sampled_indexes, predicted_col]
    target_transcriptions_sampled = df.loc[sampled_indexes, config.GROUND_TRUTH_COL]

    metrics[f'{config.WER_PREFIX}micro_average_{predicted_col_root}{suffix}'].append(
      wer_obj(predicted_transcriptions_sampled, target_transcriptions_sampled).item())
    metrics[f'{config.WER_PREFIX}macro_average_{predicted_col_root}{suffix}'].append(
      df.loc[sampled_indexes, wer_col].mean())

    metrics[f'{config.CER_PREFIX}micro_average_{predicted_col_root}{suffix}'].append(
      cer_obj(predicted_transcriptions_sampled, target_transcriptions_sampled).item())
    metrics[f'{config.CER_PREFIX}macro_average_{predicted_col_root}{suffix}'].append(
      df.loc[sampled_indexes, cer_col].mean())

    if save_indexes:
      metrics[f'sampled_indexes_{predicted_col_root}{suffix}'].append(sampled_indexes)

  return metrics


def get_confidence_interval_and_mean(means, alpha=0.95):
  mean, std = scipy.stats.norm.fit(means)
  lower, upper = scipy.stats.norm.interval(alpha, mean, std)

  return means.mean(), lower, upper


def get_confidence_interval_values_per_col(df, col_groups, within_group_names, alpha=0.95):
  mean_per_col = collections.defaultdict(list)
  lower_per_col = collections.defaultdict(list)
  upper_per_col = collections.defaultdict(list)

  for group_index, group in enumerate(col_groups):
    for col_index, within_group_name in zip(range(len(group)), within_group_names):
      mean, lower, upper = get_confidence_interval_and_mean(df[group[col_index]], alpha=alpha)

      mean_per_col[within_group_name].append(mean)
      lower_per_col[within_group_name].append(lower)
      upper_per_col[within_group_name].append(upper)

  return mean_per_col, lower_per_col, upper_per_col
