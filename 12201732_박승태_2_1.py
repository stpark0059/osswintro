import pandas as pd

data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# 2_1_1
year = [2015, 2016, 2017, 2018]  # 선택 년도

for current_year in year:
    data_year = data[data['year'] == current_year]  # 선택한 년도와 일치하는 데이터만 추출

    print(f"\n**{current_year}년도 기준**")

    for standard in ['H', 'avg', 'HR', 'OBP']:
        sorted_data = data_year.sort_values(by=[standard], ascending=False)
        top10 = sorted_data['batter_name'].head(10).reset_index(drop=True)
        print(f"{standard}: {', '.join(top10)}")

#2_1_2
data_posu = data[data['cp'] == '포수'].sort_values(by=['war'], ascending=False).head(1)
data_1ru = data[data['cp'] == '1루수'].sort_values(by=['war'], ascending=False).head(1)
data_2ru = data[data['cp'] == '2루수'].sort_values(by=['war'], ascending=False).head(1)
data_3ru = data[data['cp'] == '3루수'].sort_values(by=['war'], ascending=False).head(1)
data_ug = data[data['cp'] == '유격수'].sort_values(by=['war'], ascending=False).head(1)
data_left = data[data['cp'] == '좌익수'].sort_values(by=['war'], ascending=False).head(1)
data_center = data[data['cp'] == '중견수'].sort_values(by=['war'], ascending=False).head(1)
data_right = data[data['cp'] == '우익수'].sort_values(by=['war'], ascending=False).head(1)

print("\n**2_1_2 문제**\n\n각 포지션 별 탑 플레이어")
print("\n포수 :", data_posu['batter_name'].to_string(index=False))
print("\n1루수 :", data_1ru['batter_name'].to_string(index=False))
print("\n2루수 :", data_2ru['batter_name'].to_string(index=False))
print("\n3루수 :", data_3ru['batter_name'].to_string(index=False))
print("\n유격수 :", data_ug['batter_name'].to_string(index=False))
print("\n좌익수 :", data_left['batter_name'].to_string(index=False))
print("\n중견수 :", data_center['batter_name'].to_string(index=False))
print("\n우익수 :", data_right['batter_name'].to_string(index=False))
print("\n")

#2_1_3
columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']
need_data = data[columns]

correlations = need_data.corr()['salary']
correlations = correlations.drop('salary')
print("**2_1_3 문제**\n")
print("각 기준 별 correlation 값\n")
print(correlations)
print("\n가장 correlation이 높은 기준 :", correlations.idxmax())





    
