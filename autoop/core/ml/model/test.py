import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'Los Angeles', 'Chicago']
})

for col in df.columns:
    print(f"Column: {col}")
    print(df[col].values)

# Output:
# Column: name
# ['Alice' 'Bob' 'Charlie']
# Column: age
# [25 30 35]
# Column: city
# ['New York' 'Los Angeles' 'Chicago']