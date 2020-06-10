import pandas as pd

def find_hypothesis(data, num_classes):
  '''
    finds hypothesis using Find-S algorithm
    for all decision classes
  '''
  num_observations = data.shape[0]
  num_attributes = data.shape[1] - 1

  hypothesis = [[] for _ in range(num_classes)]

  for i in range(num_observations):
    animal_class = int(data[i][-1]) - 1

    if len(hypothesis[animal_class]) == 0:
      hypothesis[animal_class] = data[i, :-1]
    else:
      for j in range(num_attributes):
        if hypothesis[animal_class][j] != data[i][j]:
          hypothesis[animal_class][j] = '?'

  return hypothesis


if __name__ == "__main__":
  data = pd.read_csv("data/zoo.csv").applymap(str).values[:,1:]

  classes = ['Mammal', 'Bird', 'Reptile', 'Fish',
            'Amphibians', 'Insect', 'Other']

  hypothesis = find_hypothesis(data, len(classes))

  print("Hypothesis:")
  print("-" * 77)

  for i in range(len(classes)):
    indent_space = ' ' * (10 - len(classes[i]))
    print(f'{classes[i]}{indent_space}: {hypothesis[i]}')

  print("-" * 77)