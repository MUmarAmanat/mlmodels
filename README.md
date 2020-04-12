# mlmodels : Model ZOO for Pytorch, Tensorflow, Keras, Gluon, LightGBM, Sklearn models...

- Model ZOO with Lightweight Functional interface to wrap access to Recent and State of Art Deep Learning, ML models and Hyper-Parameter Search, cross platforms such as Tensorflow, Pytorch, Gluon, Keras, sklearn, light-GBM,...

- Logic follows sklearn : fit, predict, transform, metrics, save, load

- Goal is to transform Script/Research code into Re-usable/batch/ code with minimal code change ...

- Why Functional interface instead of pure OOP ?
  Functional reduces the amount of code needed, focus more on the computing part (vs design part),
  a bit easier maintenability for medium size project, good for scientific computing process.


*  Usage, Example :
https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/README_usage.md


*  Colab demo for Install :
https://colab.research.google.com/drive/1sYbrXNZh9nTeizS-AuCA8RSu94B_B-RF


## Model List :


### Time Series:
Nbeats: 2019, Time Series NNetwork,  https://arxiv.org/abs/1905.10437

Amazon Deep AR: 2019, Time Series NNetwork,  https://arxiv.org/abs/1905.10437

Facebook Prophet 2017, Time Series prediction,

ARMDN Advanced Time series Prediction :  2019, Associative and Recurrent Mixture Density Networks for time series.

LSTM prediction 



### NLP :
Sentence Transformers : 2019, Embedding of full sentences using BERT, https://arxiv.org/pdf/1908.10084.pdf

Transformers Classifier : Using Transformer for Text Classification, https://arxiv.org/abs/1905.05583

TextCNN Pytorch : 2016, Text CNN Classifier, https://arxiv.org/abs/1801.06287

TextCNN Keras : 2016, Text CNN Classifier, https://arxiv.org/abs/1801.06287

charCNN Keras : Text Character Classifier,


DRMM:  Deep Relevance Matching Model for Ad-hoc Retrieval.

DRMMTKS:  Deep Top-K Relevance Matching Model for Ad-hoc Retrieval.

ARC-I:  Convolutional Neural Network Architectures for Matching Natural Language Sentences

ARC-II:  Convolutional Neural Network Architectures for Matching Natural Language Sentences

DSSM:  Learning Deep Structured Semantic Models for Web Search using Clickthrough Data

CDSSM:  Learning Semantic Representations Using Convolutional Neural Networks for Web Search

MatchLSTM: Machine Comprehension Using Match-LSTM and Answer Pointer

DUET:  Learning to Match Using Local and Distributed Representations of Text for Web Search

KNRM:  End-to-End Neural Ad-hoc Ranking with Kernel Pooling

ConvKNRM:  Convolutional neural networks for soft-matching n-grams in ad-hoc search

ESIM:  Enhanced LSTM for Natural Language Inference

BiMPM:  Bilateral Multi-Perspective Matching for Natural Language Sentences

MatchPyramid:  Text Matching as Image Recognition

Match-SRNN:  Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN

aNMM:  aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model

MV-LSTM:  Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations

DIIN:  Natural Lanuguage Inference Over Interaction Space

HBMP:  Sentence Embeddings in NLI with Iterative Refinement Encoders




### TABULAR :
#### LightGBM

#### AutoML Gluon  :  2020, AutoML in Gluon, MxNet using LightGBM, CatBoost


#### Auto-Keras  :  2020, Automatic Keras model selection


#### All sklearn models :

linear_model.ElasticNet\
linear_model.ElasticNetCV\
linear_model.Lars\
linear_model.LarsCV\
linear_model.Lasso\
linear_model.LassoCV\
linear_model.LassoLars\
linear_model.LassoLarsCV\
linear_model.LassoLarsIC\
linear_model.OrthogonalMatchingPursuit\
linear_model.OrthogonalMatchingPursuitCV\


svm.LinearSVC\
svm.LinearSVR\
svm.NuSVC\
svm.NuSVR\
svm.OneClassSVM\
svm.SVC\
svm.SVR\
svm.l1_min_c\


neighbors.KNeighborsClassifier\
neighbors.KNeighborsRegressor\
neighbors.KNeighborsTransformer\





#### Binary Neural Prediction from tabular data:

A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |

Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |

Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |

Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |

Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |

Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |

Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |

Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |

xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |

AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |

Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |

Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |

Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |

Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |

Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |

FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |




### VISION :
  Vision Models (pre-trained) :  
alexnet\
densenet121\
densenet169\
densenet201\
densenet161\
inception_v3\
resnet18\
resnet34\
resnet50\
resnet101\
resnet152\

resnext50_32x4d\
resnext101_32x8d\

wide_resnet50_2\
wide_resnet101_2\
squeezenet1_0

squeezenet1_1\
vgg11\
vgg13\
vgg16\
vgg19\
vgg11_bn

vgg13_bn\
vgg16_bn\
vgg19_bn

googlenet\
shufflenet_v2_x0_5\
shufflenet_v2_x1_0\
mobilenet_v2

A lot more...



......

https://github.com/arita37/mlmodels/blob/dev/README_model_list.md

######################################################################################

## ① Installation

Using pre-installed online Setup :

https://github.com/arita37/mlmodels/issues/101



Manual Install as editable package in Linux

```
conda create -n py36 python=3.6.5 -y
source activate py36

cd yourfolder
git clone https://github.com/arita37/mlmodels.git mlmodels
cd mlmodels
git checkout dev
```

### Check this Colab for install :
https://colab.research.google.com/drive/1sYbrXNZh9nTeizS-AuCA8RSu94B_B-RF


##### Initialize
Will copy template, dataset, example to your folder

    ml_models --init  /yourworkingFolder/


##### To test :
    ml_optim


##### To test model fitting
    ml_models
    
        
#### Actual test runs

https://github.com/arita37/mlmodels/actions

![test_fast_linux](https://github.com/arita37/mlmodels/workflows/test_fast_linux/badge.svg)

![test_fast_windows](https://github.com/arita37/mlmodels/workflows/test_fast_windows/badge.svg?branch=dev)

![ All model testing (Linux) ](https://github.com/arita37/mlmodels/workflows/code_structure_linux/badge.svg)

#######################################################################################

## Usage in Jupyter

https://github.com/arita37/mlmodels/blob/dev/README_usage.md

#######################################################################################

## CLI tools:

https://github.com/arita37/mlmodels/blob/dev/README_usage_CLI.md



####################################################################################

## Model List

https://github.com/arita37/mlmodels/blob/dev/README_model_list.md

#######################################################################################

## How to add a new model

https://github.com/arita37/mlmodels/blob/dev/README_addmodel.md

#######################################################################################

## Index of functions/methods

https://github.com/arita37/mlmodels/blob/dev/README_index_doc.txt

####################################################################################
















### LSTM example in TensorFlow ([Example notebook](mlmodels/example/1_lstm.ipynb))

#### Define model and data definitions
```python
# import library
import mlmodels


model_uri    = "model_tf.1_lstm.py"
model_pars   =  {  "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                }
data_pars    =  {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars =  { "learning_rate": 0.001, }

out_pars     =  { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}
save_pars = { "path" : "ztest_1lstm/model/" }
load_pars = { "path" : "ztest_1lstm/model/" }



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


#### Inference
metrics_val   =  module.fit_metrics( model, sess, data_pars, compute_pars, out_pars) # get stats
ypred         = module.predict(model, sess,  data_pars, compute_pars, out_pars)     # predict pipeline


```


---

### AutoML example in Gluon ([Example notebook](mlmodels/example/gluon_automl.ipynb))
```
# import library
import mlmodels
import autogluon as ag

#### Define model and data definitions
model_uri = "model_gluon.gluon_automl.py"
data_pars = {"train": True, "uri_type": "amazon_aws", "dt_name": "Inc"}

model_pars = {"model_type": "tabular",
              "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
              "activation": ag.space.Categorical(*tuple(["relu", "softrelu", "tanh"])),
              "layers": ag.space.Categorical(
                          *tuple([[100], [1000], [200, 100], [300, 200, 100]])),
              'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
              'num_boost_round': 10,
              'num_leaves': ag.space.Int(lower=26, upper=30, default=36)
             }

compute_pars = {
    "hp_tune": True,
    "num_epochs": 10,
    "time_limits": 120,
    "num_trials": 5,
    "search_strategy": "skopt"
}

out_pars = {
    "out_path": "dataset/"
}



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars, out_pars=out_pars)      


#### Inference
ypred       = module.predict(model, data_pars, compute_pars, out_pars)     # predict pipeline


```

---

### RandomForest example in Scikit-learn ([Example notebook](mlmodels/example/sklearn.ipynb))
```
# import library
import mlmodels

#### Define model and data definitions
model_uri    = "model_sklearn.sklearn.py"

model_pars   = {"model_name":  "RandomForestClassifier", "max_depth" : 4 , "random_state":0}

data_pars    = {'mode': 'test', 'path': "../mlmodels/dataset", 'data_type' : 'pandas' }

compute_pars = {'return_pred_not': False}

out_pars    = {'path' : "../ztest"}


#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
```


---

### TextCNN example in keras ([Example notebook](example/textcnn.ipynb))

```python
# import library
import mlmodels

#### Define model and data definitions
model_uri    = "model_keras.textcnn.py"

data_pars    = {"path" : "../mlmodels/dataset/text/imdb.csv", "train": 1, "maxlen":400, "max_features": 10}

model_pars   = {"maxlen":400, "max_features": 10, "embedding_dims":50}
                       
compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"] ,
                        "batch_size": 32, "epochs":1, 'return_pred_not':False}

out_pars     = {"path": "ztest/model_keras/textcnn/"}



#### Load Parameters and Train
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
data_pars['train'] = 0
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
```

---

### Using json config file for input ([Example notebook](example/1_lstm_json.ipynb), [JSON file](mlmodels/mlmodels/example/1_lstm.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_tf.1_lstm.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/1_lstm.json'
})

#### Load parameters and train
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model

#### Check inference
ypred       = module.predict(model, sess=sess,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline


```

---

### Using Scikit-learn's SVM for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_svm.ipynb), [JSON file](mlmodels/example/sklearn_titanic_svm.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/sklearn_titanic_svm.json'
})

#### Load Parameters and Train

model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred


#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)


```

---

### Using Scikit-learn's Random Forest for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_randomForest.ipynb), [JSON file](mlmodels/example/sklearn_titanic_randomForest.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/example/sklearn_titanic_randomForest.json'
})


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model


#### Inference

ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred

#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)

```

---

### Using Autogluon for Titanic Problem from json file ([Example notebook](mlmodels/example/gluon_automl_titanic.ipynb), [JSON file](mlmodels/example/gluon_automl.json))

#### Import library and functions
```python
# import library
import mlmodels

#### Load model and data definitions from json
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_gluon.gluon_automl.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(
    choice='json',
    config_mode= 'test',
    data_path= '../mlmodels/example/gluon_automl.json'
)


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model   =  module.fit(model, model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
model.model.fit_summary()


#### Check inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline

#### Check metrics
model.model.model_performance

import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)


```

---
---

### Using hyper-params (optuna) for Titanic Problem from json file ([Example notebook](mlmodels/example/sklearn_titanic_randomForest_example2.ipynb), [JSON file](mlmodels/example/hyper_titanic_randomForest.json))

#### Import library and functions
```python
# import library
from mlmodels.models import module_load
from mlmodels.optim import optim
from mlmodels.util import params_json_load


#### Load model and data definitions from json

###  hypermodel_pars, model_pars, ....
model_uri   = "model_sklearn.sklearn.py"
config_path = path_norm( 'example/hyper_titanic_randomForest.json'  )
config_mode = "test"  ### test/prod



#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


module            =  module_load( model_uri= model_uri )                      
model_pars_update = optim(
    model_uri       = model_uri,
    hypermodel_pars = hypermodel_pars,
    model_pars      = model_pars,
    data_pars       = data_pars,
    compute_pars    = compute_pars,
    out_pars        = out_pars
)


#### Load Parameters and Train
model         =  module.Model(model_pars=model_pars_update, data_pars=data_pars, compute_pars=compute_pars)y
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)

#### Check inference
ypred         = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred


#### Check metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv( path_norm('dataset/tabular/titanic_train_preprocessed.csv') )
y = y['Survived'].values
roc_auc_score(y, ypred)


```


---

### Using LightGBM for Titanic Problem from json file ([Example notebook](mlmodels/example/model_lightgbm.ipynb), [JSON file](mlmodels/example/lightgbm_titanic.json))

#### Import library and functions
```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm
import json

#### Load model and data definitions from json
# Model defination
model_uri    = "model_sklearn.model_lightgbm.py"
module        =  module_load( model_uri= model_uri)

# Path to JSON
data_path = '../dataset/json/lightgbm_titanic.json'  

# Model Parameters
pars = json.load(open( data_path , mode='r'))
for key, pdict in  pars.items() :
  globals()[key] = path_norm_dict( pdict   )   ###Normalize path

#### Load Parameters and Train
model = module.Model(model_pars, data_pars, compute_pars) # create model instance
model, session = module.fit(model, data_pars, compute_pars, out_pars) # fit model


#### Check inference
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # get predictions
ypred


#### Check metrics
metrics_val = module.fit_metrics(model, data_pars, compute_pars, out_pars)
metrics_val 

```

---




### Using Vision CNN RESNET18 for MNIST dataset  ([Example notebook](mlmodels/example/model_restnet18.ipynb), [JSON file](mlmodels/model_tch/torchhub_cnn.json))

```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
import json


#### Model URI and Config JSON
model_uri   = "model_tch.torchhub.py"
config_path = path_norm( 'model_tch/torchhub_cnn.json'  )
config_mode = "test"  ### test/prod





#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 
`
#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)




```
---



### Using ARMDN Time Series : Ass for MNIST dataset  ([Example notebook](mlmodels/example/model_timeseries_armdn.ipynb), [JSON file](mlmodels/model_keras/armdn.json))



```python
# import library
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
import json


#### Model URI and Config JSON
model_uri   = "model_keras.ardmn.py"
config_path = path_norm( 'model_keras/ardmn.json'  )
config_mode = "test"  ### test/prod




#### Model Parameters
hypermodel_pars, model_pars, data_pars, compute_pars, out_pars = params_json_load(config_path, config_mode= config_mode)
print( hypermodel_pars, model_pars, data_pars, compute_pars, out_pars)


#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 
`
#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           #### fit model
metrics_val    = module.fit_metrics(model, data_pars, compute_pars, out_pars)   #### Check fit metrics
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)



#### Save/Load
module.save(model, save_pars ={ 'path': out_pars['path'] +"/model/"})

model2 = module.load(load_pars ={ 'path': out_pars['path'] +"/model/"})



```
---



