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
  filtered_data = data[np.array([data[i][filter_column] == filter_value for i in range(num_rows)])]
  return filtered_data


def gain(data, columns, column):
  '''
    calculates gain of data 
    after taking column
  '''
  column_counts = {}
  num_rows = data.shape[0]
  gain = entropy(data)

  for i in range(num_rows):
    try:
      column_counts[data[i][column]] += 1
    except:
      column_counts[data[i][column]] = 1

  for key in column_counts.keys():
    filtered_data = filter_data(data, column, key)
    gain -= (column_counts[key] / num_rows) * entropy(filtered_data)

  return gain


def decision_tree_build(data, columns, column, possible_values, remaining_columns, type, rule):
  '''
    generates decision tree rules
    for given data using ID3 algo
  '''
  num_columns = data.shape[1]

  if type == 0:
    temp_entropy = entropy(data)

    if temp_entropy == 0:
      rule.append(data[0][-1])
      print(rule)
      rule.pop()
      return

    max_col = mx = -1
    for _,i in enumerate(remaining_columns):
      temp = gain(data, columns, i)
      if temp > mx:
        mx = temp
        max_col = i

    decision_tree_build(data, columns, max_col, possible_values, remaining_columns, 1, rule)

  else:
    
    remaining_columns.remove(column)
    for _,i in enumerate(possible_values[columns[column]]):
      filtered_data = filter_data(data, column, i)
      rule.append(f'{columns[column]} = {i}')
      decision_tree_build(filtered_data, columns, column, possible_values, remaining_columns, 0, rule)
      rule.pop()


if __name__ == "__main__":
  data = pd.read_csv("data/ID3_data.csv").values
  columns = ["Outlook","Temp","Humidity","Wind","Play Tennis?"]
  remaining_columns = set([x for x in range(len(columns) - 1)])

  possible_values = extract_possible_values(data, columns)

  print("Decision Tree Rules: ")
  decision_tree_build(data, columns, 0, possible_values, remaining_columns, 0, [])