from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']:
    if font in fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
SEED = 2026

def preprocessing(frame):
    categorical = frame.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numerical = [c for c in frame if c not in categorical]
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True), numerical),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='Unknown')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ])

def save_submission(sample, ids, target, prediction):
    assert list(sample.columns) == [ids.name, target], '제출 열 확인 필요'
    assert len(ids) == len(prediction) == len(sample)
    assert sample[ids.name].astype(str).tolist() == ids.astype(str).tolist(), '제출 ID 순서 불일치'
    assert np.isfinite(prediction).all()
    result = sample.copy()
    result[target] = prediction
    result.to_csv(OUT / 'submission.csv', index=False)
    return result

# %% 원본 파일 기간과 사건번호 중복 검증
files = sorted(Path('한국소비자원_소비자_피해규제').glob('*.csv'))
assert files, '원본 CSV가 없습니다.'
frames, audit = [], []
for path in files:
    data = pd.read_csv(path, encoding='cp949')
    data.columns = data.columns.str.strip()
    dates = pd.to_datetime(data['접수일(년월일)'], errors='coerce')
    audit.append({'file': path.name, 'rows': len(data), 'start': str(dates.min()), 'end': str(dates.max())})
    data['_source'] = path.name
    frames.append(data)
raw = pd.concat(frames, ignore_index=True)
display(pd.DataFrame(audit))
pd.DataFrame(audit).to_csv(OUT / 'file_audit.csv', index=False)
KEY = '사건번호'
assert raw[KEY].notna().all()
cols = [c for c in raw.columns if c != '_source']
duplicate_ids = raw.loc[raw[KEY].duplicated(False), cols]
if len(duplicate_ids):
    conflicts = duplicate_ids.groupby(KEY).nunique(dropna=False).gt(1).any(axis=1)
    assert not conflicts.any(), '같은 사건번호에 서로 다른 값이 있어 수동 확인이 필요합니다.'
df = raw.drop_duplicates(KEY).copy()
df['접수일'] = pd.to_datetime(df['접수일(년월일)'], errors='coerce')
assert df['접수일'].notna().all()
assert df['접수일'].between('2023-01-01', '2024-12-31').all()
df['연도'] = df['접수일'].dt.year
df['월'] = df['접수일'].dt.month
df['연월'] = df['접수일'].dt.to_period('M').astype(str)
display(df[cols].isna().sum().rename('결측 수').to_frame())
print('원본 행:', len(raw), '중복 제거:', len(raw)-len(df), '분석 사건:', len(df))
# %% 건수와 비율. 결측도 별도 범주로 표시.
summary_metrics = {'raw_rows': len(raw), 'duplicate_rows_removed': len(raw)-len(df), 'unique_cases': len(df)}
for col in ['성별', '연령대', '지역', '판매유형', '물품소분류', '청구이유']:
    counts = df[col].fillna('(결측)').value_counts(dropna=False)
    table = counts.rename('건수').to_frame()
    table['비율(%)'] = table['건수'] / len(df) * 100
    assert table['건수'].sum() == len(df)
    display(table.head(15))
    table.to_csv(OUT / f'counts_{col}.csv')
    if col == '성별':
        summary_metrics['gender_counts'] = counts.to_dict()
        summary_metrics['female_minus_male_cases'] = int(counts.get('여자', 0) - counts.get('남자', 0))
    ax = table.head(10)['건수'].sort_values().plot.barh(figsize=(9, 5), title=f'{col}별 접수 사건 수 (상위 10개)')
    ax.set(xlabel='접수 사건 수', ylabel=''); plt.tight_layout(); plt.savefig(OUT / f'counts_{col}.png'); plt.show(); plt.close()
# %% 연도별 월 추이와 품목 내 청구이유 구성
monthly = df.groupby(['연도', '월']).size().unstack('연도').reindex(range(1, 13), fill_value=0)
display(monthly)
monthly.to_csv(OUT / 'monthly_by_year.csv')
ax = monthly.plot(marker='o', figsize=(10, 5), title='연도별 월별 피해구제 접수 사건 수')
ax.set(xlabel='월', ylabel='접수 사건 수'); plt.xticks(range(1, 13)); plt.tight_layout(); plt.savefig(OUT / 'monthly_by_year.png'); plt.show(); plt.close()
top_products = df['물품소분류'].value_counts().head(10).index
product_reasons = pd.crosstab(df['물품소분류'].fillna('(결측)'), df['청구이유'].fillna('(결측)')).loc[top_products]
reason_share = product_reasons.div(product_reasons.sum(axis=1), axis=0)*100
display(reason_share.round(1))
reason_share.to_csv(OUT / 'reason_share_within_product.csv')
assert np.allclose(reason_share.sum(axis=1), 100)
# %% 관찰 사실과 가설 구분
summary_metrics['period'] = [str(df['접수일'].min().date()), str(df['접수일'].max().date())]
(OUT / 'metrics.json').write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary_metrics, ensure_ascii=False, indent=2))
print('관찰: 접수 데이터의 규모와 구성, 연도별 월 변화 및 품목 내 청구이유 비중을 확인했다.')
print('가설: 계절별 거래량이나 이용량이 접수 건수 차이에 영향을 줄 수 있다. 외부 거래량 없이 원인을 확정할 수 없다.')
print('제안: 접수 상위 품목의 주요 청구이유를 기준으로 상담 안내와 대응 인력 배치를 검토한다.')
print('한계: 접수 사건만 포함한다. 인구·거래량 분모, 해결 여부·처리 기간이 없어 피해 발생률과 해결 효율은 추정하지 않는다.')
