import pandas as pd
from sklearn.model_selection import train_test_split

seed = 0

data = pd.read_csv('./data/data.tsv', delimiter='\t', encoding='utf-8')
print('In the dataset, there are {} subjective sentences and {} objective sentences.'
      .format(*data['label'].value_counts()))

train, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['label'])
print('In the test data, there are {} subjective sentences and {} objective sentences.'
      .format(*test['label'].value_counts()))

train, val = train_test_split(train, test_size=0.2, random_state=seed, stratify=train['label'])
print('In the training data, there are {} subjective sentences and {} objective sentences.'
      .format(*train['label'].value_counts()))
print('In the validation data, there are {} subjective sentences and {} objective sentences.'
      .format(*val['label'].value_counts()))

train.to_csv('./data/train.tsv', sep='\t', index=False)
val.to_csv('./data/validation.tsv', sep='\t', index=False)
test.to_csv('./data/test.tsv', sep='\t', index=False)
