from utils.argparser import parse_cmd_args
from modes.train import train
from modes.test import test
from modes.train_encoder import train_encoder
from data.data_transform import process_files
from utils.time_keeper import time_keeper, time_function


def main():

    args = parse_cmd_args()

    if args.mode == "test":
        test(args)
    else:
        train(args)

    time_keeper.summary()


# just to run the encoder quickly
def main_encoder():
    args = parse_cmd_args()
    process_files()
    train_encoder(args)


if __name__ == "__main__":
    main()
