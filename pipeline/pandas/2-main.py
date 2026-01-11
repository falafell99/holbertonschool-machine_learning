#!/usr/bin/env python3
from_file = __import__('2-from_file').from_file

# Create test CSV
import pandas as pd
test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
test_data.to_csv('test.csv', index=False)

df = from_file('test.csv', ',')
print(df)
