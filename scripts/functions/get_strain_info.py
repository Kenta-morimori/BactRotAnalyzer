import os

from . import param


def main(day):
    input_dir = f"{param.input_dir_bef}{day}"
    strain_list = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    return strain_list
