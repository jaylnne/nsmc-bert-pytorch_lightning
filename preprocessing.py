import re
import pandas as pd
from tqdm import tqdm


def removing_non_korean(df):
    for idx, row in tqdm(df.iterrows(), desc='removing_non_korean', total=len(df)):
        new_doc = re.sub('[^가-힣]', '', row['document']).strip()
        df.loc[idx, 'document'] = new_doc
    return df


def generate_preprocessed(data_path):

    train = pd.read_csv(f'{data_path}/ratings_train.txt', sep='\t')
    test = pd.read_csv(f'{data_path}/ratings_test.txt', sep='\t')

    # 필요없는 열은 drop
    train.drop(['id'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)
    
    # null 제거
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    
    # 중복 제거
    train.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)
    test.drop_duplicates(subset=['document'], inplace=True, ignore_index=True)
    
    train.to_csv(f'{data_path}/train_clean.csv', index=False)
    test.to_csv(f'{data_path}/test_clean.csv', index=False)

    # 한국어 외 제거
    train = removing_non_korean(train)
    test = removing_non_korean(test)
    
    train.to_csv(f'{data_path}/train_only_korean.csv', index=False)
    test.to_csv(f'{data_path}/test_only_korean.csv', index=False)
    
    # mecab, komoran 형태소 분석 결과는 테스트 성능이 낮아 포함하지 않음