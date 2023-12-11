import itertools

import numpy as np
import scipy.stats

from matplotlib import pyplot as plt

import bisk_metrics
import bisk_preprocessing


def plot_counts_per_gender(df, gender_col, title, figsize=(4, 4), ylim=None, rotation=0):
  ax = df[gender_col].value_counts().plot(kind='bar', title=title, figsize=figsize)

  text_boxes = ax.bar_label(ax.containers[0], padding=2)
  for text_box in text_boxes:
    text_box.set_backgroundcolor('white')
    text_box.set_bbox({'boxstyle': 'square,pad=0.1', 'fc': 'white'})

  ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
  if ylim is not None:
    ax.set_ylim(ax.get_ylim()[0], ylim)
  else:
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.05)

  return ax


def plot_word_counts(df, sentence_col, title, figsize=(20, 10), ylim=None, rotation=0):
  ax = bisk_preprocessing.get_word_count(df, sentence_col).plot(
    kind='bar', title=title, figsize=figsize)

  text_boxes = ax.bar_label(ax.containers[0], padding=2)
  for text_box in text_boxes:
    text_box.set_backgroundcolor('white')
    text_box.set_bbox({'boxstyle': 'square,pad=0.1', 'fc': 'white'})

  ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
  if ylim is not None:
    ax.set_ylim(ax.get_ylim()[0], ylim)
  else:
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.05)

  return ax


def plot_bars(
      df,
      affixes_to_match,
      affixes_for_splitting,
      group_names,
      within_group_names,
      title,
      ylabel,
      ax=None,
      within_group_name_filter=None,
      alpha=0.95,
      figsize=(15, 4),
      x_label_rotation=0,
      ylim=None,
      colors=None,
      custom_grid=False,
      bar_label_font_size='medium',
):
  cols = _get_filtered_cols(df, affixes_to_match)

  affixes_as_funcs = _get_affixes_as_funcs(affixes_for_splitting)

  col_groups = _get_col_groups(cols, affixes_as_funcs)

  mean_per_col, lower_per_col, upper_per_col = bisk_metrics.get_confidence_interval_values_per_col(
      df, col_groups, within_group_names, alpha=alpha,
  )

  return _plot_bars(
      df, group_names, within_group_names, within_group_name_filter,
      mean_per_col, lower_per_col, upper_per_col, title, ylabel,
      figsize, ax,
      x_label_rotation=x_label_rotation, ylim=ylim, colors=colors,
      custom_grid=custom_grid, bar_label_font_size=bar_label_font_size)


def _plot_bars(
        df, group_names, within_group_names, within_group_name_filter,
        mean_per_col, lower_per_col, upper_per_col, title, ylabel,
        figsize, ax,
        x_label_rotation=0, bar_width=0.25, ylim=None, colors=None, custom_grid=False,
        bar_label_font_size='medium'):
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
  else:
    fig = ax.get_figure()

  x = np.arange(len(group_names))
  bar_width = bar_width

  if within_group_name_filter is None:
    within_group_name_filter = list(range(len(within_group_names)))

  mean_per_col_list = list(mean_per_col.items())

  mean_per_col_filtered = {}
  for i in within_group_name_filter:
    name, values = mean_per_col_list[i]
    mean_per_col_filtered[name] = values

  if colors is not None:
    colors_filtered = [color for i, color in enumerate(colors) if i in within_group_name_filter]
  else:
    colors_filtered = [None] * len(within_group_name_filter)

  for index, ((within_group_name, values), color) in (
        enumerate(zip(mean_per_col_filtered.items(), colors_filtered))):
    offset = bar_width * index

    lowers = np.array(mean_per_col[within_group_name]) - np.array(lower_per_col[within_group_name])
    uppers = np.array(upper_per_col[within_group_name]) - np.array(mean_per_col[within_group_name])

    rects = ax.bar(
        x + offset,
        values,
        bar_width,
        label=within_group_name,
        yerr=[lowers, uppers],
        color=color,
        error_kw=dict(elinewidth=1, capsize=3.0, alpha=0.65))
    text_boxes = ax.bar_label(rects, padding=4, fmt='%.2f')
    for text_box in text_boxes:
      text_box.set_fontsize(bar_label_font_size)
      text_box.set_backgroundcolor('white')
      text_box.set_bbox({'boxstyle': 'square,pad=0.0', 'facecolor': 'white', 'edgecolor': 'none'})

  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.set_xticks(x + (bar_width * len(mean_per_col_filtered) / 2 - bar_width / 2), group_names)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=x_label_rotation)
  ax.legend()

  if ylim is None:
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.05)
  else:
    ax.set_ylim(ax.get_ylim()[0], ylim)

  if custom_grid:
    ax.grid(axis='y', c='lightgrey', linewidth=0.2)

  return fig, ax


def _get_filtered_cols(df, affixes_to_match):
  return [col for col in df.columns if all(affix in col for affix in affixes_to_match)]


def _get_col_groups(cols, affixes_as_funcs):
  col_groups = []

  for match_col_funcs in itertools.product(*affixes_as_funcs):
    cols_per_group = []

    for col in cols:
      if all(func(col) for func in match_col_funcs):
        cols_per_group.append(col)

    col_groups.append(cols_per_group)

  return col_groups


def _get_affixes_as_funcs(affixes_for_splitting):
  affixes_as_funcs = []
  for affix_group in affixes_for_splitting:
    affix_func_group = []

    for affix in affix_group:
      if callable(affix):
        affix_func_group.append(affix)
      else:
        affix_func_group.append(lambda col, affix_=affix: affix_ in col)

    affixes_as_funcs.append(affix_func_group)

  return affixes_as_funcs
