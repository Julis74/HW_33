from datetime import datetime
import os
import dill
import pandas as pd

from glob import glob


path = os.environ.get('PROJECT_PATH', '.')


list_of_files = []
for pkl in glob(f'{path}/data/models/*.pkl'):
    list_of_files.append(pkl)

latest_file = max(list_of_files, key=os.path.getctime)
name_of_pkl = latest_file.split("\\")[1]

with open(f'{path}/data/models/{name_of_pkl}', 'rb') as file:
    model = dill.load(file)


def predict():
    import json
    prediction_df = pd.DataFrame()
    for datapath in glob(f'{path}/data/test/*.json'):
        with open(datapath, 'rb') as datafile:
            df = pd.json_normalize(json.load(datafile))
            y = model.predict(df)
            prediction_df = pd.concat([prediction_df, pd.DataFrame({'id': df['id'], 'price': df['price'], 'prediction': y[0]})], ignore_index=True)
    prediction_df.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
