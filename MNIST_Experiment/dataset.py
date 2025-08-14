from mindspore.dataset import MnistDataset
import os
import mindspore as ms
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

# 数据集生成函数
# def create_dataset(data_dir, training=True, batch_size=32, resize=(28, 28),
#                    rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=64):
#     # 生成训练集和测试集的路径
#     data_train = os.path.join(data_dir, 'train') # train set
#     data_test = os.path.join(data_dir, 'test') # test set
#     # 利用MnistDataset方法读取mnist数据集，如果training是True则读取训练集
#     ds = ms.dataset.MnistDataset(data_train if training else data_test)
#     # map方法是非常有效的方法，可以整体对数据集进行处理，r
#     # esize改变数据形状，
#     # rescale进行归一化，
#     # HWC2CHW改变图像通道
#     ds = ds.map(input_columns=["image"], operations=[
#         CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
#     #利用map方法改变数据集标签的数据类型
#     ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))
#     # shuffle是打乱操作，同时设定了batchsize的大小，并将最后不足一个batch的数据抛弃
#     ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

#     return ds


def create_dataset(data_dir,training=True,batch_size=32,resize=(224, 224),
                   rescale=1 / (255 * 0.3081),shift=-0.1307 / 0.3081,buffer_size=64):
    #生成路径
    data_dir = os.path.join(data_dir)
    #利用Cifar10Dataset方法读cifar10数据集，如果training是True则读取训练集
    ds = ms.dataset.Cifar10Dataset(dataset_dir=data_dir,
                                   usage='train' if training else 'test')
    # map方法是非常有效的方法，
    # 可以整体对数据集进行处理，
    # resize改变数据形状，
    # rescale进行归一化，H
    # WC2CHW改变图像通道
    ds = ds.map(input_columns=["image"],
                operations=[
                    CV.Resize(resize),
                    CV.Rescale(rescale, shift),
                    CV.HWC2CHW()
                ])
    #利用map方法改变数据集标签的数据类型
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))
    # shuffle是打乱操作，同时设定了batchsize的大小，并将最后不足一个batch的数据抛弃
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size,
                                                   drop_remainder=True)

    return ds

