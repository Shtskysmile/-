from utils import train
import argparse
import mindspore.context as context

def main(args):
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    num_epochs = args.num_epochs
    train(data_dir='./data',
          lr=lr,
          momentum=momentum,
          num_epochs=num_epochs,
          batch_size=batch_size)

#main函数负责调用之前定义的函数，完成整个训练验证过程
if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='GPU')  #  CPU, GPU
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--num_epochs", default=10)
    args = parser.parse_args()
    main(args)
