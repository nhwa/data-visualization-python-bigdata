# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from datetime import datetime

# 한글 표현을 위한 설정 
from matplotlib import font_manager, rc
#font_name = font_manager.FontProperties(fname="D:/practice/NanumBarunGothic.ttf").get_name()
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

input_file_korea = './한국관광객카드사용데이터csv.csv'
data_frame_korea = pd.read_csv(input_file_korea, encoding='CP949')
data_frame_korea.columns = data_frame_korea.columns.str.replace('\s+', '')

# 1) 국내 관광객 지역,월별 카드이용량 => 어느 지역에서 어느 시기에 소비가 많이 일어나는 지 판단

지역별카드이용금액 = data_frame_korea.filter(['기준년월','제주대분류','제주중분류','카드이용금액'])
지역별카드이용금액['기준년월'] = pd.to_datetime(지역별카드이용금액['기준년월'])
지역별카드이용금액['기준년월'] = 지역별카드이용금액['기준년월'].dt.month
지역별카드이용금액['지역'] = np.where(지역별카드이용금액['제주중분류'].str.contains('동'), 지역별카드이용금액['제주대분류'] + ' 도심',
                           지역별카드이용금액['제주대분류'] + ' 도심 외')

sum_by_카드이용금액_제주시도심 = 지역별카드이용금액[지역별카드이용금액['지역'] == '제주시 도심'].groupby('기준년월').카드이용금액.sum()
sum_by_카드이용금액_제주시도심외 = 지역별카드이용금액[지역별카드이용금액['지역'] == '제주시 도심 외'].groupby('기준년월').카드이용금액.sum()
sum_by_카드이용금액_서귀포시도심 = 지역별카드이용금액[지역별카드이용금액['지역'] == '서귀포시 도심'].groupby('기준년월').카드이용금액.sum()
sum_by_카드이용금액_서귀포시도심외 = 지역별카드이용금액[지역별카드이용금액['지역'] == '서귀포시 도심 외'].groupby('기준년월').카드이용금액.sum()

label = ['1월', '2월', '3월', '4월','5월','6월','7월','8월','9월','10월','11월','12월']
N = len(지역별카드이용금액['기준년월'].unique())
bar_width = 0.1
index = np.arange(N)
alpha = 0.5

p1 = plt.bar(index, sum_by_카드이용금액_제주시도심, 
             bar_width, 
             color='b', 
             alpha=alpha,
             label='제주시 도심')

p2 = plt.bar(index + bar_width, sum_by_카드이용금액_제주시도심외, 
             bar_width, 
             color='r', 
             alpha=alpha,
             label='제주시 도심 외')

p3 = plt.bar(index + bar_width + bar_width, sum_by_카드이용금액_서귀포시도심, 
             bar_width, 
             color='y', 
             alpha=alpha,
             label='서귀포시 도심')

p4 = plt.bar(index + bar_width + bar_width + bar_width, sum_by_카드이용금액_서귀포시도심외, 
             bar_width, 
             color='g', 
             alpha=alpha,
             label='서귀포시 도심 외')

plt.title('2020년 국내 관광객 지역, 월별 카드이용금액', fontsize=20)
plt.ylabel('카드이용금액', fontsize=18)
plt.xlabel('월', fontsize=18)
plt.xticks(index, label, fontsize=15)
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('제주시 도심', '제주시 도심 외','서귀포시 도심','서귀포시 도심 외'), fontsize=15)
plt.show()

fig = plt.figure()

지역별카드이용건수 = data_frame_korea.filter(['기준년월','제주대분류','제주중분류','카드이용건수'])
지역별카드이용건수['기준년월'] = pd.to_datetime(지역별카드이용건수['기준년월'])
지역별카드이용건수['기준년월'] = 지역별카드이용건수['기준년월'].dt.month
지역별카드이용건수['지역'] = np.where(지역별카드이용건수['제주중분류'].str.contains('동'), 지역별카드이용건수['제주대분류'] + ' 도심',
                           지역별카드이용건수['제주대분류'] + ' 도심 외')

sum_by_카드이용건수_제주시도심 = 지역별카드이용건수[지역별카드이용건수['지역'] == '제주시 도심'].groupby('기준년월').카드이용건수.sum()
sum_by_카드이용건수_제주시도심외 = 지역별카드이용건수[지역별카드이용건수['지역'] == '제주시 도심 외'].groupby('기준년월').카드이용건수.sum()
sum_by_카드이용건수_서귀포시도심 = 지역별카드이용건수[지역별카드이용건수['지역'] == '서귀포시 도심'].groupby('기준년월').카드이용건수.sum()
sum_by_카드이용건수_서귀포시도심외 = 지역별카드이용건수[지역별카드이용건수['지역'] == '서귀포시 도심 외'].groupby('기준년월').카드이용건수.sum()

label = ['1월', '2월', '3월', '4월','5월','6월','7월','8월','9월','10월','11월','12월']
N = len(지역별카드이용건수['기준년월'].unique())
bar_width = 0.1
index = np.arange(N)
alpha = 0.5

p1 = plt.bar(index, sum_by_카드이용건수_제주시도심, 
             bar_width, 
             color='b', 
             alpha=alpha,
             label='제주시 도심')

p2 = plt.bar(index + bar_width, sum_by_카드이용건수_제주시도심외, 
             bar_width, 
             color='r', 
             alpha=alpha,
             label='제주시 도심 외')

p3 = plt.bar(index + bar_width + bar_width, sum_by_카드이용건수_서귀포시도심, 
             bar_width, 
             color='y', 
             alpha=alpha,
             label='서귀포시 도심')

p4 = plt.bar(index + bar_width + bar_width + bar_width, sum_by_카드이용건수_서귀포시도심외, 
             bar_width, 
             color='g', 
             alpha=alpha,
             label='서귀포시 도심 외')

plt.title('2020년 국내 관광객 지역, 월별 카드이용건수', fontsize=20)
plt.ylabel('카드이용건수', fontsize=18)
plt.xlabel('월', fontsize=18)
plt.xticks(index, label, fontsize=15)
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('제주시도심', '제주시도심외','서귀포시도심','서귀포시도심외'), fontsize=15)
plt.show()

월별카드이용금액 = data_frame_korea.filter(['기준년월','카드이용금액'])
월별카드이용금액['기준년월'] = pd.to_datetime(월별카드이용금액['기준년월'])
월별카드이용금액['월'] = 월별카드이용금액['기준년월'].dt.month
월별카드이용금액df= 월별카드이용금액.groupby(['월']).sum()
월별카드이용금액df.plot(kind='line', linestyle='--', marker=r'*', alpha=1.0, title="2020년 국내 관광객 월별 카드이용금액", color=u'green')
plt.show()

# 2) 국내 관광객 연령대별, 성별 카드이용량 => 어느 연령대, 성별에서 소비가 많이 일어나는 지 판단
fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]

연령대별카드이용금액 = data_frame_korea.filter(['연령대별','성별','카드이용금액'])
연령대별카드이용건수 = data_frame_korea.filter(['연령대별','성별','카드이용건수'])

sum_by_카드이용금액_성별 = pd.DataFrame(연령대별카드이용금액.groupby(['연령대별','성별']).카드이용금액.sum())
sum_by_카드이용금액_성별 =sum_by_카드이용금액_성별.reset_index()
sum_by_카드이용건수_성별 = pd.DataFrame(연령대별카드이용건수.groupby(['연령대별','성별']).카드이용건수.sum())
sum_by_카드이용건수_성별 =sum_by_카드이용건수_성별.reset_index()

sum_by_카드이용금액_성별_pivot = sum_by_카드이용금액_성별.pivot(index='연령대별', columns='성별', values='카드이용금액')
sum_by_카드이용건수_성별_pivot = sum_by_카드이용건수_성별.pivot(index='연령대별', columns='성별', values='카드이용건수')

fig.suptitle('2020년 국내 관광객 연령대별 카드이용량', fontsize=18)
sum_by_카드이용금액_성별_pivot.plot(kind='barh', ax=ax1, stacked=True, rot=0,title="카드이용금액")
plt.setp(ax1.get_xticklabels(), rotation=45, fontsize=12)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=12)
ax1.set_xlabel('카드이용금액')
ax1.yaxis.set_ticks_position('left')

sum_by_카드이용건수_성별_pivot.plot(kind='barh', ax=ax2,stacked=True, rot=0,title="카드이용건수")
plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=12)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=12)
ax2.set_xlabel('카드이용건수')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]

fig.suptitle('2020년 국내 관광객 성별 카드이용량', fontsize=18)
성별카드이용금액 = data_frame_korea.filter(['성별','카드이용금액'])
성별카드이용횟수 = data_frame_korea.filter(['성별','카드이용건수'])

성별카드이용금액df= 성별카드이용금액.groupby(['성별']).sum()
성별카드이용금액df.plot(kind='bar', ax=ax1, alpha=1.0, title="카드이용금액")
plt.setp(ax1.get_xticklabels(), rotation=45, fontsize=10)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=10)
ax1.set_xlabel('성별')
ax1.set_ylabel('카드이용금액')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

성별카드이용횟수df= 성별카드이용횟수.groupby(['성별']).sum()
성별카드이용횟수df.plot(kind='bar', ax=ax2, alpha=1.0, title="카드이용건수", color='dodgerblue')
plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=10)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=10)
ax2.set_xlabel('성별')
ax2.set_ylabel('카드이용건수')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('right')
plt.show()


# 3) 국내 관광객 업종별 카드이용량 => 어느 업종에서 소비가 많이 일어나는 지 판단
fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]

fig.suptitle('2020년 국내 관광객 업종별 카드이용량', fontsize=18)
업종별카드이용금액 = data_frame_korea.filter(['업종명','카드이용금액'])
업종별카드이용횟수 = data_frame_korea.filter(['업종명','카드이용건수'])

업종별카드이용금액df= 업종별카드이용금액.groupby(['업종명']).sum()
업종별카드이용금액df.plot(kind='bar', ax=ax1, alpha=1.0, title="카드이용금액")
plt.setp(ax1.get_xticklabels(), rotation=45, fontsize=10)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=10)
ax1.set_xlabel('업종명')
ax1.set_ylabel('카드이용금액')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

업종별카드이용횟수df= 업종별카드이용횟수.groupby(['업종명']).sum()
업종별카드이용횟수df.plot(kind='bar', ax=ax2, alpha=1.0, title="카드이용건수", color='dodgerblue')
plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=10)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=10)
ax2.set_xlabel('업종명')
ax2.set_ylabel('카드이용건수')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('right')
plt.show()

# 중국 관광객 
input_file_china = '중국관광객카드사용데이터csv.csv'
data_frame_china = pd.read_csv(input_file_china, encoding='CP949')
data_frame_china.columns = data_frame_china.columns.str.replace('\s+', '')

# 4) 중국 관광객 지역별 카드이용량 => 어느 지역에서 소비가 많이 일어나는 지 판단
fig, axes = plt.subplots(nrows=1, ncols=3)
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]

fig.suptitle('2014.09 - 2016.08 중국 관광객 지역별 카드이용량', fontsize=18)

data_frame_china['카드이용금액'] = data_frame_china['2014-09~2015-08카드이용금액'] + data_frame_china['2015-09~2016-08카드이용금액']
data_frame_china['카드이용건수'] = data_frame_china['2014-09~2015-08카드이용건수'] + data_frame_china['2015-09~2016-08카드이용건수']
data_frame_china['카드이용자수'] = data_frame_china['2014-09~2015-08카드이용자수'] + data_frame_china['2015-09~2016-08카드이용자수']

data_frame_china['지역'] = np.where(data_frame_china['지역구분_시'].str.contains('전체'),data_frame_china['지역구분_시'], 
                           data_frame_china['지역구분_시']+' '+data_frame_china['지역구분_도심/도심외'])

지역별카드이용금액 = data_frame_china.filter(['지역','카드이용금액'])
지역별카드이용건수 = data_frame_china.filter(['지역','카드이용건수'])
지역별카드이용자수 = data_frame_china.filter(['지역','카드이용자수'])

sum_by_카드이용금액 = 지역별카드이용금액.groupby(['지역']).카드이용금액.sum()
sum_by_카드이용건수 = 지역별카드이용건수.groupby(['지역']).카드이용건수.sum()
sum_by_카드이용자수 = 지역별카드이용자수.groupby(['지역']).카드이용자수.sum()

my_labels=['서귀포시 도심','서귀포시 도심 외','전체','제주시 도심', '제주시 도심 외']

ax1.set_title('카드이용금액')
ax2.set_title('카드이용건수')
ax3.set_title('카드이용자수')

sum_by_카드이용금액.plot(kind='pie', ax=ax1, fontsize=13, labels=my_labels, subplots=True, y="카드이용금액" ,autopct='%1.1f%%')
sum_by_카드이용건수.plot(kind='pie', ax=ax2, fontsize=13, labels=my_labels, subplots=True, y="카드이용건수" ,autopct='%1.1f%%')
sum_by_카드이용자수.plot(kind='pie', ax=ax3 ,fontsize=13, labels=my_labels, subplots=True, y="카드이용자수" ,autopct='%1.1f%%')

plt.show()

fig = plt.figure()
업종별카드이용금액 = data_frame_china.filter(['관광업종구분','지역','2015-09~2016-08카드이용금액'])
업종별카드이용금액df= 업종별카드이용금액.groupby(['지역','관광업종구분']).sum()
업종별카드이용금액df.rename(columns = {"2015-09~2016-08카드이용금액": "2015 - 2016 카드이용금액"}, inplace = True)

업종별카드이용금액df = 업종별카드이용금액df.reset_index()
step = 업종별카드이용금액df.pivot('관광업종구분','지역','2015 - 2016 카드이용금액') 
                  # 열인덱스, 행인덱스, 데이터    순서로 들어감
sns.heatmap(step)
plt.title('중국 관광객 지역별 카드이용금액', fontsize=18)
plt.show() 

# 5) 중국 관광객 기간별 카드이용금액

fig = plt.figure()
업종별카드이용금액 = data_frame_china.filter(['관광업종구분','2014-09~2015-08카드이용금액','2015-09~2016-08카드이용금액'])
업종별카드이용금액df= 업종별카드이용금액.groupby(['관광업종구분']).sum()
업종별카드이용금액df.rename(columns = {"2014-09~2015-08카드이용금액": "2014 - 2015 카드이용금액","2015-09~2016-08카드이용금액": "2015 - 2016 카드이용금액"}, inplace = True)
sns.heatmap(업종별카드이용금액df)
plt.title('중국 관광객 기간별 카드이용금액', fontsize=18)
plt.show() 


#6) 국내 관광객과 중국 관광객의 제주시 도심의 업종별 매출 현황 비교
fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]

fig.suptitle('국내 관광객과 중국 관광객별 도심 업종 매출 비율', fontsize=18)

data_frame_korea['지역'] = np.where(data_frame_korea['제주중분류'].str.contains('동'), data_frame_korea['제주대분류'] + ' 도심',
                           data_frame_korea['제주대분류'] + ' 도심 외')
국내관광객도심업종별카드이용금액 = data_frame_korea.filter(['지역','업종명','카드이용금액'])
국내관광객도심업종별카드이용금액df = 국내관광객도심업종별카드이용금액[국내관광객도심업종별카드이용금액['지역'] == '제주시 도심'].groupby('업종명').카드이용금액.sum()
국내관광객도심업종별카드이용금액df['식음료'] = 국내관광객도심업종별카드이용금액df['슈퍼 마켓'] + 국내관광객도심업종별카드이용금액df['기타음료식품'] +국내관광객도심업종별카드이용금액df['스넥']   
국내관광객도심업종별카드이용금액df['의류'] = 국내관광객도심업종별카드이용금액df['정장'] + 국내관광객도심업종별카드이용금액df['정장(여성)']   
국내관광객도심업종별카드이용금액df.drop(['슈퍼 마켓','스넥','기타음료식품'],inplace=True)
국내관광객도심업종별카드이용금액df.drop(['정장','정장(여성)'],inplace=True)
# 0%대 항목은 그래프의 가시성을 위해 삭제
국내관광객도심업종별카드이용금액df.drop(['골프 용품','귀 금 속','신   발','악세 사리'],inplace=True)

data_frame_china['지역'] = np.where(data_frame_china['지역구분_시'].str.contains('전체'),data_frame_china['지역구분_시'], 
                           data_frame_china['지역구분_시']+' '+data_frame_china['지역구분_도심/도심외'])
중국관광객도심업종별카드이용금액 = data_frame_china.filter(['지역','관광업종구분','2015-09~2016-08카드이용금액'])
중국관광객도심업종별카드이용금액.rename(columns = {"2015-09~2016-08카드이용금액": "카드이용금액"}, inplace = True)
중국관광객도심업종별카드이용금액df = 중국관광객도심업종별카드이용금액[국내관광객도심업종별카드이용금액['지역'] == '제주시 도심'].groupby('관광업종구분').카드이용금액.sum()
# 0% 항목은 그래프의 가시성을 위해 삭제
중국관광객도심업종별카드이용금액df.drop(['교통','유흥'],inplace=True)

ax1.set_title('국내관광객')
ax2.set_title('중국관광객')

wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

국내관광객도심업종별카드이용금액df.plot(kind='pie', ax=ax1, fontsize=13, startangle=200, subplots=True, y="카드이용금액" ,autopct='%0.1f%%',counterclock=False, wedgeprops=wedgeprops, explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
중국관광객도심업종별카드이용금액df.plot(kind='pie', ax=ax2, fontsize=13, startangle=200, subplots=True, y="카드이용금액" ,autopct='%0.1f%%',counterclock=False, wedgeprops=wedgeprops, explode = [0.05, 0.05, 0.05, 0.05, 0.05])

plt.show()
