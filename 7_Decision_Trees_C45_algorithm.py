import math
import numpy as np
import pandas as pd


def extract_possible_values(data, columns):
  '''
    extracts all the possible of 
    each column of data and stores it 
    in a dictionary
  '''
  num_columns = data.shape[1]
  possible_values = {}

  for i in range(num_columns):
    possible_values[columns[i]] = set(data[:, i])

  return possible_values


def entropy(data):
  '''
    calculates entropy of given data
  '''
  num_rows = data.shape[0]

  counts = {}
  for i in range(num_rows):
    try:
      counts[data[i][-1]] += 1
    except:
      counts[data[i][-1]] = 1
  
  entropy = 0
  for value in counts.values():
    entropy -= (value / num_rows) * math.log2(value / num_rows)

  return entropy


def filter_data(data, filter_column, filter_value):
  '''
    filters the data on given 
    column and value
  '''
  num_rows = data.shape[0]
  
  # extracts rows with filter column value = filter value
  filtered_data = data[np.array([data[i][filter_column] == filter_value for i in range(num_rows)])]

  return filtered_data


def filter_continous_data(data, filter_column, threshold, less_than):
  '''
    filters the data on given
    column and threshold
  '''
  num_rows = data.shape[0]
  filtered_data = []

  if less_than:
    # extracts rows with filter column value <= threshold
    filtered_data = data[np.array(
        [data[i][filter_column] <= threshold for i in range(num_rows)])]
  else:
    # extracts rows with filter column value > threshold
    filtered_data = data[np.array(
        [data[i][filter_column] > threshold for i in range(num_rows)])]
  
  return filtered_data


def split_ratio(data, columns, column):
  '''
    calculates split ratio of 
    categorical column
  '''
  column_counts = {}
  num_rows = data.shape[0]
  split_ratio = 0

  for i in range(num_rows):
    try:
      column_counts[data[i][column]] += 1
    except:
      column_counts[data[i][column]] = 1

  for key in column_counts.keys():
    split_ratio -= (column_counts[key] / num_rows) * math.log2(column_counts[key] / num_rows)

  return split_ratio


def split_ratio_continous(data, columns, column, threshold):
  '''
    calculates split ratio of
    continous column
  '''
  column_counts = {}
  num_rows = data.shape[0]
  split_ratio = 0

  lt = gt = 0

  for i in range(num_rows):
    if data[i][column] <= threshold:
      lt += 1
    else:
      gt += 1

  if lt > 0:
    split_ratio -= (lt / num_rows) * math.log2(lt / num_rows)

  if gt > 0:
    split_ratio -= (gt / num_rows) * math.log2(gt / num_rows)

  return split_ratio


def gain_continous(data, columns, column, threshold):
  '''
    calculates gain of data
    after taking column (continous)
  '''
  num_rows = data.shape[0]
  column_gain = entropy(data)
  lt = gt = 0

  for i in range(num_rows):
    if data[i][column] <= threshold:
      lt += 1
    else:
      gt += 1

  column_gain -= ( lt / num_rows ) * entropy(filter_continous_data(data, column, threshold, True))

  column_gain -= ( gt / num_rows ) * entropy(filter_continous_data(data, column, threshold, False))

  return column_gain

def gain(data, columns, column):
  '''
    calculates gain of data 
    after taking column (categorical)
  '''
  column_counts = {}
  num_rows = data.shape[0]
  column_gain = entropy(data)

  for i in range(num_rows):
    try:
      column_counts[data[i][column]] += 1
    except:
      column_counts[data[i][column]] = 1

  for key in column_counts.keys():
    filtered_data = filter_data(data, column, key)
    column_gain -= (column_counts[key] / num_rows) * entropy(filtered_data)

  return column_gain


def gain_ratio(data, columns, column, possible_values):
  '''
    calculates gain ratio of data
    after taking column 

    if continous data column, also
    finds best threshold
  '''
  if not type(data[0][column]) is int:
    # if column is categorical valued
    column_gain = gain(data, columns, column)
    split = split_ratio(data, columns, column)

    return (column_gain / split)
  
  else:
    # if column is continous valued
    best_threshold = mx = mx_split = -1

    # try every possible threshold (ignoring the last two values)
    for i, threshold in enumerate(possible_values[columns[column]]):
     
      
      if i >= len(possible_values[columns[column]]) - 2:
        break

      split = split_ratio_continous(data, columns, column, threshold)
      column_gain = gain_continous(data, columns, column, threshold)

      if not split:
        continue

      temp_gain = column_gain / split

      # if gain better than current max gain, mark it as best threshold
      if column_gain >= mx:
        mx = column_gain
        best_threshold = threshold 
        mx_split = split 
    
    return mx / mx_split, best_threshold


def decision_tree_build(data, columns, column, possible_values, remaining_columns, level_type, threshold, rule):
  '''
    generates decision tree rules
    for given data using ID3 algo
  '''
  num_columns = data.shape[1]

  if level_type == 0:
    temp_entropy = entropy(data)

    if temp_entropy == 0:
      # leaf node
      rule.append(data[0][-1])
      print(rule)
      rule.pop()
      return

    max_col = mx = best_threshold = -1

    for _,i in enumerate(remaining_columns):      
      if type(data[0][i]) is int:
        # if column is continous valued, find best threshold and gain
        temp, temp_threshold = gain_ratio(data, columns, i, possible_values)
        
        if temp > mx:
          best_threshold = temp_threshold

      else:
        # if column is categorical valued
        temp = gain_ratio(data, columns, i, possible_values)

      # if column has better gain, mark it as best
      if temp > mx:
        mx = temp
        max_col = i

    decision_tree_build(data, columns, max_col, possible_values, remaining_columns, 1, best_threshold, rule)

  else:
    
    remaining_columns.remove(column)

    if type(data[0][column]) is int:
      # if column is continous valued

      # call for value <= best threshold
      filtered_data = filter_continous_data(data, column, threshold, True)

      rule.append(f'{columns[column]} <= {threshold}')

      decision_tree_build(filtered_data, columns, column, possible_values, remaining_columns, 0, threshold, rule)

      rule.pop()

      # call for value > best threshold
      filtered_data = filter_continous_data(data, column, threshold, False)

      rule.append(f'{columns[column]} > {threshold}')

      decision_tree_build(filtered_data, columns, column, possible_values, remaining_columns, 0, threshold, rule)

      rule.pop()

    else:
      # if column is categorical

      for _,i in enumerate(possible_values[columns[column]]):
        
        # call for every possible value
        filtered_data = filter_data(data, column, i)

        rule.append(f'{columns[column]} = {i}')

        decision_tree_build(filtered_data, columns, column, possible_values, remaining_columns, 0, threshold, rule)

        rule.pop()


if __name__ == "__main__":
  data = pd.read_csv("data/C45_data.csv")

  columns = list(data.columns.values)
  remaining_columns = set([x for x in range(len(columns) - 1)])
  # remaining_columns.remove(1)

  data = data.values
  possible_values = extract_possible_values(data, columns)

  print("Decision Tree Rules: ")
  decision_tree_build(data, columns, 0, possible_values, remaining_columns, 0, 0, [])
