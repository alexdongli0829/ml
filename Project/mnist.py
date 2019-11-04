import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from collections import deque

train_data = pd.read_csv("/home/ubuntu/gluon_tutorials/mnist_data/train.csv")
test_data = pd.read_csv("/home/ubuntu/gluon_tutorials/mnist_data/test.csv")

X = train_data.drop('label',axis=1)
y = train_data['label']

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.feature_selection import SelectFromModel

#x_train, x_valid,y_train, y_valid = train_test_split(X, y, test_size=0.1)

x_train, y_train = X, y

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
import mxnet as mx
import datetime


x_train_pic = x_train.values.reshape(-1,28,28,1)
#x_valid_pic = x_valid.values.reshape(-1,28,28,1)
train_dataset = gdata.dataset.ArrayDataset(nd.array(x_train_pic), nd.array(y_train.values))
#valid_dataset = gdata.dataset.ArrayDataset(nd.array(x_valid_pic), nd.array(y_valid.values))

def get_net(ctx):
    finetune_net= model_zoo.vision.resnet18_v2()
    finetune_net.output = nn.Dense(10)
    # need to put the parameters on to model cpu or gpu()
    #finetune_net.features.initialize(init.Xavier(), ctx=ctx)
    #finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net

#If sparse_label is True (default), label should contain integer category indicators:
loss = gloss.SoftmaxCrossEntropyLoss()

def get_loss(data, net, ctx):
    l=0
    for X, y in data:
        output=net(X.as_in_context(ctx))
        l += loss(output, y.as_in_context(ctx)).mean().asscalar()
    return l/len(data)

#image transformation, resize, crop, flip and nomorlize
transform_train = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(40),
    gdata.vision.transforms.RandomResizedCrop(28, scale=(0.64,1), ratio = (1.0,1.0)),
    #gdata.vision.transforms.RandomFlipLeftRight(),
    #for ToTensor, check this: https://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html#ToTensor
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914],
                                      [0.2023])
])
transform_valid= gdata.vision.transforms.Compose([
    #gdata.vision.transforms.Resize(40),
    #gdata.vision.transforms.RandomResizedCrop(32, scale=(0.64,1), ratio = (1.0,1.0)),
    #gdata.vision.transforms.RandomFlipLeftRight(),
    #for ToTensor, check this: https://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html#ToTensor
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914],
                                      [0.2023])
])

def train(net, train_set, valid_set, lr, wd, num_epochs, lr_decay,lr_period, batch_size, ctx):
    trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate':lr, 'momentum':0.9, 'wd': wd})
    previous_time=datetime.datetime.now()
    train_data = gdata.DataLoader(train_set.transform_first(transform_train), batch_size=batch_size)
    if valid_set is not None:
    	valid_data = gdata.DataLoader(valid_set.transform_first(transform_valid), batch_size=batch_size)

    val_loss = 1
    current_los = 1
    q = deque()
    for epoch in range(num_epochs):
        train_l = 0
        if epoch > 0 and epoch%lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        for X, y in train_data:
            with autograd.record():
                output = net(X.as_in_context(ctx))
                #l = loss(output,y.astype('float32').as_in_context(ctx))
                l = loss(output,y.as_in_context(ctx))
            l.backward()
            #print(l)
            train_l += l.mean().asscalar()
            trainer.step(batch_size)

        current_time=datetime.datetime.now()
        h,reminder=divmod((current_time-previous_time).seconds, 3600)
        m,s = divmod(reminder, 60)
        time_s = "time: %02d:%02d:%02d" %(h,m,s)
        if valid_set is not None:
                current_loss = get_loss(valid_data, net, ctx)
                tmp=val_loss - current_loss
                val_loss = current_loss 
                q.append(tmp)
                improv = 0
                #check the lastest 5 iterates, if the improvment is less than 0.0002, then breach the epochs loop
                if len(q) > 5:
                        q.popleft()
                        for i in q:
                            improv = improv + i
                        if improv < 0.00001:
                                print("performance not improvment a lot, quite the loop, result:")
                                print(q)
                                break
        else:
                current_loss = train_l/len(train_data)
                tmp=val_loss - current_loss
                val_loss = current_loss
                q.append(tmp)
                improv = 0
                #check the lastest 5 iterates, if the improvment is less than 0.0002, then breach the epochs loop
                if len(q) > 5:
                        q.popleft()
                        for i in q:
                            improv = improv + i
                        if improv < 0.00001:
                                print("performance not improvment a lot, quite the loop, result:")
                                print(q)
                                break
        if epoch % 1 == 0:
            if valid_set is not None:  
                print("epoch %d, train_loss %f, valid_loss %f, learning_rate %f, time %s"%
                      (epoch, train_l/len(train_data), current_loss, 
                      trainer.learning_rate, time_s))
            else:
                print("epoch %d, train_loss %f, learning_rate %f, time %s"%
                      (epoch, train_l/len(train_data),trainer.learning_rate,time_s))

	

ctx, num_epochs, lr, wd, batch_size = mx.gpu(0), 1000, 0.0001, 5e-4, 50
lr_period, lr_decay, net = 15, 0.5, get_net(ctx)
net.hybridize()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
train(net, train_dataset, None, lr, wd, num_epochs, 
      lr_decay, lr_period, batch_size, ctx)	
net.save_parameters("resnet18_predict")
