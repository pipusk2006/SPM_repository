import pandas as pd

# Загрузка файла с ошибками
fail_df = pd.read_csv("fail.csv")

# Вывод средних значений по числовым признакам
print("Средние значения по признакам для False Positive (ошибочно предсказанный инсульт):")
print(fail_df.median(numeric_only=True))
