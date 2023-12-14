import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Oil Temperature Time Series Forecasting')
    parser.add_argument('--input_size', type=int, default=7, help='input size')
    parser.add_argument('--output_size', type=int, default=7, help='output size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--O', type=int, default=96, help='predict observation length, 96 or 336')
    parser.add_argument('--model', type=str, default='transformer', help='model name', choices=['lstm', 'transformer'])
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    return parser.parse_args()


args = parse_args()
