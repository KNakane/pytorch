import argparse
import torch
from Utility.trainer import Trainer
from Dataset.dataset import Dataset
from Network.MLP import MLP
from Network.CNN import LeNet

def main(args):

    dataset = Dataset(name=args.data,
                      batch_size=args.batch_size,)

    model = eval(args.network)(input_height=dataset.height,
                               input_width=dataset.width,
                               input_channel=dataset.channel,
                               output_shape=dataset.class_num)

    trainer = Trainer(model=model,
                      dataset=dataset,
                      epoch=args.n_epoch,
                      optimizer=args.opt,
                      learning_rate=args.lr,
                      save_checkpoint_steps=args.save_checkpoint_steps,
                      checkpoints_to_keep=args.checkpoints_to_keep,
                      debug=args.debug)

    trainer()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--network', default='MLP', type=str,
        choices=['MLP', 'LeNet', 'VGG', 'GoogLeNet', 'ResNet18', 'ResNet34', 'DenseNet', 'VisionTransformer',
                 'AutoEncoder', 'VAE', 'CVAE', 'AAE',
                 'GAN', 'DCGAN', 'WGAN', 'WGANGP', 'LSGAN', 'ACGAN', 'BEGAN', 'SNGAN',
                 'RNN', 'LSTM', 'GRU', 'FCN', 'TCN']
    )
    parser.add_argument('--data', default='mnist', type=str, choices=['mnist', 'cifar10', 'cifar100', 'kuzushiji'])
    parser.add_argument('--n_epoch', default=1000, type=int, help='Input max epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Input learning rate')
    parser.add_argument('--opt', default='SGD', type=str,
                        choices=['SGD', 'Momentum', 'Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'AdaBound', 'AMSGrad',
                                 'AdaBelief'])
    parser.add_argument('--aug', default=None, type=str,
                        choices=['shift', 'mirror', 'rotate', 'shift_rotate', 'cutout', 'random_erace'])
    parser.add_argument('--denoise', action='store_true', help='True : Denoising AE, False : standard AE')
    parser.add_argument('--conv', action='store_true', help='True : Convolutional AE, False : standard AE')
    parser.add_argument('--l2_norm', action='store_true', help='L2 normalization or not')
    parser.add_argument('--z_dim', default=100, type=int, help='Latent z dimension')
    parser.add_argument('--conditional', action='store_true', help='Conditional true or false')
    parser.add_argument('--n_disc_update', default=1, type=int, help='Learning times for discriminator')
    parser.add_argument('--init_model', default=None, type=str,
                        help='Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    parser.add_argument('--checkpoints_to_keep', default=5, type=int, help='checkpoint keep count')
    parser.add_argument('--keep_checkpoint_every_n_hours', default=1, type=int, help='checkpoint create hour')
    parser.add_argument('--save_checkpoint_steps', default=5, type=int, help='save checkpoint step')
    parser.add_argument('--debug', action='store_true', help='DEBUG MODE')
    args = parser.parse_args()
    main(args)