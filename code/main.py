from utils.argparser import parse_cmd_args
from modes.train import train
from modes.visualize import visualize
from modes.test import test

def main():
    
    args = parse_cmd_args()

    if args.visualize:
        visualize(args)
    elif args.test:
        test(args)
    else: 
        train(args)
        

if __name__ == '__main__':
    main()
