from dataset import create_dataset
from model import MLP, LeNet, LeNet5
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from log import Logger
from datetime import datetime
from mindspore.common.initializer import XavierUniform  # 导入 Xavier 初始化器

# 构建训练、验证函数进行模型训练和验证，提供数据路径，设定学习率，epoch数量
def train(data_dir, lr=0.001, momentum=0.9, num_epochs=10, batch_size=32):
    logger = Logger()
    start_time = datetime.now()
    #调用函数，读取训练集
    ds_train = create_dataset(data_dir, batch_size=batch_size)
    #调用函数，读取验证集
    ds_eval = create_dataset(data_dir, training=False)
    #构建网络
    net = MLP()
    # net = LeNet()
    # net = LeNet5()
    print(net)
    #设定loss函数
    loss = nn.loss.CrossEntropyLoss(reduction='mean')
    #设定优化器
    # opt = nn.Momentum(net.trainable_params(), lr, momentum)  # 原Momentum优化器
    opt = nn.Adam(net.trainable_params(), learning_rate=lr)  # 使用Adam优化器
    # opt = nn.SGD(net.trainable_params(), learning_rate=lr, momentum=momentum)
    #设定损失监控
    loss_cb = LossMonitor(per_print_times=ds_train.get_dataset_size())
    #编译形成模型
    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    # 训练网络，dataset_sink_mode为on_device模式
    model.train(num_epochs,
                ds_train,
                callbacks=[loss_cb],
                dataset_sink_mode=False)
    #用验证机评估网络表现
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    #输出相关指标
    end_time = datetime.now()
    print('Metrics:', metrics)
    print('Training time:', end_time - start_time)
    #将相关指标写入log文件
    logger.write({'net': str(net)})
    logger.write({'opt': str(opt)})
    logger.write({'loss': str(loss)})
    logger.write({'lr': lr})
    logger.write({'momentum': momentum})
    logger.write({'num_epochs': num_epochs})
    logger.write({'batch_size': batch_size})
    logger.write({'metrics': metrics})
    logger.write({'training_time': (end_time - start_time).total_seconds()})
    logger.write({'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S')})
    logger.write({'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')})
