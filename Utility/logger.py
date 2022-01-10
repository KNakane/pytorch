import os, sys
import datetime
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG, INFO

def get_logger(outdir="./results/", debug=False):
    """

    :param outdir:
    :param debug:
    :return:
    """
    # dirの設定
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filename = os.path.join(outdir, "logging.log")

    # logの設定
    logger = getLogger("Pytorch")
    logger = _set_handler(logger, StreamHandler(), debug)
    logger = _set_handler(logger, FileHandler(filename), debug)
    logger.setLevel(DEBUG if debug else INFO)
    logger.propagate = False
    return logger


def _set_handler(logger, handler, verbose):
    if verbose:
        handler.setLevel(DEBUG)
    else:
        handler.setLevel(INFO)
    handler.setFormatter(Formatter('%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s'))
    logger.addHandler(handler)
    return logger
