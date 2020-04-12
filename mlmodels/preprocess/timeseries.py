"""

  Ensemble of preprocessor for time series

"""

from mlmodels.util import path_norm
import pandas as pd





def pd_load(path) :
   return pd.read_csv(path_norm(path ))




def pd_clean(df,  pars={}) :
  df = df.fillna(0)
  return df



def pd_reshape(test, features, target, pred_len, m_feat) :
    x_test = test[features]
    x_test = x_test.values.reshape(-1, pred_len, m_feat)
    y_test = test[target]
    y_test = y_test.values.reshape(-1, pred_len, 1)        
    return x_test, y_test



def time_train_test_split(data_pars):
    """
       train_data_path
       test_data_path
       predict_only

    """
    d = data_pars
    pred_len = d["prediction_length"]
    features = d["col_Xinput"]
    target   = d["col_ytarget"]
    m_feat   = len(features)


    # when train and test both are provided
    if d["test_data_path"]:
        test   = pd_load(d["test_data_path"])
        test   = pd_clean(test)
        x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
        if d["predict_only"]:
            return x_test, y_test


        train   = pd_load( d["train_data_path"])
        train   = pd_clean(train)
        x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 

        return x_train, y_train, x_test, y_test
    


    # for when only train is provided
    df      = pd_load(d["train_data_path"])
    train   = df.iloc[:-pred_len]
    train   = pd_clean(train)
    x_train, y_train = pd_reshape(train, features, target, pred_len, m_feat) 


    test   = df.iloc[-pred_len:]
    test   = pd_clean(test)
    x_test, y_test = pd_reshape(test, features, target, pred_len, m_feat) 
    if d["predict_only"]:
        return x_test, y_test

    return x_train, y_train, x_test, y_test


