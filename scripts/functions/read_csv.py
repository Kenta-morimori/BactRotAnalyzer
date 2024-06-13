import pandas as pd

from . import param, read_csv


def read_center_coordinates(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/center_coordinate.csv"
    sample_num, _, _ = param.get_config(day)

    df = pd.read_csv(csv_save_dir)
    center_x_list = df.filter(regex=f'No.\d+_x').iloc[0, :sample_num].tolist()
    center_y_list = df.filter(regex=f'No.\d+_y').iloc[0, :sample_num].tolist()
    
    return center_x_list, center_y_list

