import pandas as pd
from ydata_profiling import ProfileReport
df = pd.read_csv('data/merged_sales_data.csv')
profile = ProfileReport(df)
profile.to_file('data/report.html')
