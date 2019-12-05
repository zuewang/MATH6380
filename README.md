# MATH 6380O
## Project 1
### preprocess
* usage example
```
dataloader = DataLoader('/path/to/directory/of/semi-conductor-image-classification-first')
train_ratio = 0.7
data = dataloader.get_data(train_ratio)
print(data['train_data'].shape)
print(data['validation_data'].shape)
print(data['test_data'].shape)
print(data['train_labels'].shape)
print(data['validation_labels'].shape)
```