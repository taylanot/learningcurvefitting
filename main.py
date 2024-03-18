# Import external 
from sacred import Experiment 
# Import local
from model import *
from fit import *
from data import *

ex = Experiment('my_experiment',save_git_info=False)

@ex.config
def my_config():
    """ 
        Configuration goes here everything is arranged as a nested dictionary.
    """
    seed = 24 # KOBEEEE

    conf = dict()
    conf["what"] = "HELLO"

def echo(what):
    """ 
        This is a simple function to capture.
    """
    print(what)

@ex.automain
def run(conf):
    """ 
        Main experiment runs in this one.
    """
    echo(conf["what"])

if __name__ == '__main__':
    ex.run()
