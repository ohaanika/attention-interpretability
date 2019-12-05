import argparse
import os


def combine_files():
    with open(os.path.join(ARGS.dataset, ARGS.dataset+'.txt'), 'w') as outfile:
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(ARGS.dataset, ARGS.dataset+'-'+split+'.txt')) as infile:
                outfile.write(infile.read())


def main(ARGS):
    """Main body of the code"""
    # combine original dataset files into one
    combine_files()


def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--dataset',  
                        type=str,  
                        choices=['IMDB', 'yelp'],  
                        default='IMDB',  
                        help='Select dataset')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    main(ARGS)