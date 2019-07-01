import sys

from main import runFitter


if __name__ == "__main__":

    # get config file and path to it
    config_file, experiment_dir = sys.argv[1], sys.argv[2]

    X, Y, Etotdict, mlonR, mlatR, Iokdict, divEdict, Imaskdict, label = runFitter(config_file,experiment_dir)


