# MATH 6380O
## Project 1
### preprocess
* usage
```
dataloader = DataLoader('/path/to/directory/of/images')
dataloader.transform()
data = dataloader.get_data()
print(data['train_data'].shape)
print(data['validation_data'].shape)
print(data['test_data'].shape)
print(data['train_labels'].shape)
print(data['validation_labels'].shape)
```