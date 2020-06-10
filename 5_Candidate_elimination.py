import numpy as np
import pandas as pd

def generalize_S(row, S):
  '''
    generalizes S wrt row of data 
  '''
  for i in range(len(S)):
    if S[i] == 'phi':
      S[i] = row[i]
    elif S[i] != row[i]:
      S[i] = '?'


def remove_inconsistent_G(S, G):
  '''
    removes hypotheses which are 
    inconsistent with S
  '''
  for j in range(len(S)):

    for i,hypothesis in enumerate(G):
      if hypothesis[j] != '?' and S[j] != hypothesis[j]:
        del G[i]


def hypothesis_satisfies_row(row, hypothesis):
  '''
    checks if current row of data satisfies the hypothesis
  '''
  for i in range(len(row) - 1):
    if hypothesis[i] != row[i] and hypothesis[i] != '?':
      return False

  return True


def specialize_G(data, row, new_hypothesis, hypothesis, S):
  '''
    specializes G wrt row of data
  '''
  temp = []

  for i in range(len(row) - 1):
    if hypothesis[i] != '?':
      continue
    
    if S[i] == 'phi':
      possible_values = data[data.columns[i]].unique()
      to_be_added = possible_values[possible_values != row[i]]
      
      for value in to_be_added:
        temp = hypothesis.copy()
        temp[i] = value
        new_hypothesis.append(temp)

    elif S[i] != '?' and S[i] != row[i]:
      temp = hypothesis.copy()
      temp[i] = S[i]
      new_hypothesis.append(temp)


def print_S_and_G(S, G):
  '''
    prints S and G in a stylized form
  '''
  temp = ' '.join(S)
  print(f'\nS: {temp}')

  print(f'G: ', end = '')
  for i, hypothesis in enumerate(G):
    temp = ' '.join(hypothesis)

    if i != 0:
      temp = '   ' + temp
      
    print(temp)


def candidate_elimination(data):
  '''
    finds general and specific hypotheses for data
  '''
  num_columns = data.shape[1]

  S = ['phi' for _ in range(num_columns - 1)]
  G = [['?' for _ in range(num_columns - 1)]]

  print_S_and_G(S, G)

  for row in data.values:

    if row[-1] == 'Y':
      generalize_S(row, S)
      remove_inconsistent_G(S, G)

    else:
      new_hypothesis = []

      for i,hypothesis in enumerate(G):
        if hypothesis_satisfies_row(row, hypothesis):
          specialize_G(data, row, new_hypothesis, hypothesis, S)
          del G[i]

      G.extend(new_hypothesis)

    print_S_and_G(S, G)


if __name__ == "__main__":
  data = pd.read_csv("data/enjoysports.csv").applymap(str)

  candidate_elimination(data)
