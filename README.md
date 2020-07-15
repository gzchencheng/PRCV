基于 fastai 的糖尿病视网膜病变检测

背景：糖尿病性视网膜病变 ( diabetic etinopathy,DR ) 是糖尿病的眼部严重并发症，随着糖尿病患者的日趋增多，DR 已成为目前特别 是发达国家 20~74 岁成年人致盲的首要原因。本项目通过fastai下的卷积神经网络模型squeezenet对训练集和验证集的眼底图片进行训练，并对测试集数据进行DR分级、DR分诊与病灶检测。

平台：python3.7+pytorch+fastai

具体过程及结果请查看该目录下的PDF文档

具体代码见该目录下的.py文件

数据集下载地址：

训练集：http://lixirong.net/teaching/prcv20/dr_train.zip 

验证集：http://lixirong.net/teaching/prcv20/dr_val.zip 

测试集：http://lixirong.net/teaching/prcv20/dr_test_no_labels.zip
