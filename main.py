import os
import sys
from loguru import logger
from argparse import ArgumentParser

from macrec.tasks import *

def main():
    init_parser = ArgumentParser()
    init_parser.add_argument('-m', '--main', type=str, required=True, help='The main function to run')
    init_parser.add_argument('--verbose', type=str, default='INFO', choices=['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'], help='The log level')
    init_args, init_extras = init_parser.parse_known_args()

    logger.remove()
    # Terminal output: only show high-level messages (SUCCESS, WARNING, ERROR, CRITICAL)
    # But allow DEBUG/INFO if explicitly requested via --verbose
    terminal_level = init_args.verbose if init_args.verbose in ['DEBUG', 'TRACE'] else 'SUCCESS'
    logger.add(sys.stderr, level=terminal_level)
    os.makedirs('logs', exist_ok=True)
    # Log file: keep all detailed INFO level logging including prompts and token usage
    logger.add('logs/{time:YYYY-MM-DD_HH-mm-ss}.log', level='INFO')

    try:
        task = eval(init_args.main + 'Task')()
    except NameError:
        logger.error('No such task!')
        return
    task.launch()

if __name__ == '__main__':
    main()
