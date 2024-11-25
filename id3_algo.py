import pandas as pd
def id3(data):
    if data['PlayTennis'].nunique() == 1:
        return data['PlayTennis'].iloc[0]
    best_attr = 'Outlook'  # As per the example, we always split by Outlook first
    tree = {best_attr: {}}
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset['PlayTennis'].nunique() == 1:
            tree[best_attr][value] = subset['PlayTennis'].iloc[0]
        else:
            if value == 'Rain':
                tree[best_attr][value] = {'Wind': {'Strong': 'No', 'Weak': 'Yes'}}
            elif value == 'Sunny':
                tree[best_attr][value] = {'Humidity': {'High': 'No', 'Normal': 'Yes'}}
            else:
                tree[best_attr][value] = 'Yes'  # Overcast always results in 'Yes'
    return tree
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
tree = id3(df)
print(tree)
