from itsdangerous import TimestampSigner
import pandas as pd
from ailib import *
from autoviz.classify_method import data_cleaning_suggestions ,data_suggestions
from autoviz.AutoViz_Class import AutoViz_Class
from ailib.ml.models.cls_xgb_large import ModelParam, Model
from datetime import datetime

train_df = pd.read_csv('/home/wandikai/work/testspace/playground-series-s3e1/train.csv')
test_df = pd.read_csv('/home/wandikai/work/testspace/playground-series-s3e1/test.csv')

train_X = train_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
        'AveOccup', 'Latitude', 'Longitude']].values
train_y = train_df['MedHouseVal'].values

test_X = test_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
        'AveOccup', 'Latitude', 'Longitude']].values

train_idx, valid_idx = train_test_split(range(len(train_X)), train_size=0.8)

train_Xy = xgb.DMatrix(train_X[train_idx], train_y[train_idx])
valid_Xy = xgb.DMatrix(train_X[valid_idx], train_y[valid_idx])
test_Xy = xgb.DMatrix(test_X, test_X)

eval_set = [(train_Xy, "train"),(valid_Xy, "test")]
model_param = ModelParam()
model_param['objective'] = 'reg:squarederror'
model_param['eval_metric'] = ['error', 'logloss', 'rmse']
model_config = model_param.to_config()

config_args = {
    "nthread":model_config.nthread,
    "eta":model_config.learning_rate,
    "gamma":model_config.gamma,
    "max_depth":model_config.max_depth,
    "min_child_weight":model_config.min_child_weight,
    "subsample":model_config.subsample,
    "colsample_bytree":model_config.colsample_bytree,
    "reg_lambda":model_config.reg_lambda,
    "reg_alpha":model_config.reg_alpha,
    "early_stopping_rounds":model_config.early_stopping_rounds,
    "eval_metric":model_config.eval_metric,
    "tree_method":model_config.tree_method,
    "scale_pos_weight":model_config.scale_pos_weight,
    "objective":model_config.objective
}
evals_result = {}

model = xgb.train(params=config_args, dtrain=train_Xy, num_boost_round=model_config.n_estimators,
            evals=eval_set, obj=model_config.obj, early_stopping_rounds = model_config.early_stopping_rounds,
            evals_result=evals_result, verbose_eval=model_config.verbose)

submission_df = pd.DataFrame({'id':test_df['id'], 'MedHouseVal':model.predict(test_Xy)})
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
output_path = Path(f'output/{timestamp}')
output_path.mkdir(parents=True, exist_ok=True)
model.save_model(output_path/'model.pt')
submission_df.to_csv(output_path/'submission.csv', index=False)
