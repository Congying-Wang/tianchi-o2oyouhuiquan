import sys
sys.path.append("../")
import tensorflow as tf

from utils.config import process_config
from utils.utils import get_args
from utils.dirs import create_dirs

from data_loader.data_generator import DataGenerator

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()
    
    # create data generator
    data = DataGenerator(config)

if __name__ == '__main__':
    main()
