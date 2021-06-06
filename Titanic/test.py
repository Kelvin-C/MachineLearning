from lib import read_csv, transpose

headers, data = read_csv('./csv_data/train.csv', 12)
data = transpose(data)

names = data[3]
names_parts = transpose([name.split(',') for name in names])
surnames = [surname.strip() for surname in names_parts[0]]
surnames_set = set(surnames)
print(len(surnames_set))

titles = [name.split('.')[0].strip() for name in names_parts[1]]
titles_set = set(titles)

title_dict = '{'
for i, title in enumerate(titles_set):
    title_dict += f"'{title}': {i+1}, "
title_dict += '}'
print(title_dict)

