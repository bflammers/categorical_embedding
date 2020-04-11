
import json

from typing import Union

import pandas as pd


def read_json(path: str) -> Union[list, dict]:

    """ Read a json file """

    with open(path, 'r') as infile:
        x = json.load(infile)

    return x


def write_json(x: Union[list, dict], path: str):

    """ Write a dict or list to a json file """

    with open(path, 'w') as outfile:
        json.dump(x, outfile)


def to_DataFrame(x: Union[list, dict]) -> pd.DataFrame:

    """ Convert a list (with row-wise dicts) or dict to a pandas DataFrame """

    # Convert data to pandas DataFrame
    if isinstance(x, pd.DataFrame):
        df = x
    elif isinstance(x, dict):
        df = pd.DataFrame(x, index=[0])
    elif isinstance(x, list):
        df = pd.DataFrame(x)
    else:
        raise NotImplementedError('Input type {} not supported'
                                  .format(type(x)))

    return df


def to_dict(x: pd.DataFrame, drop_na: bool = True) -> Union[list, dict]:

    """ Convert a pandas DataFrame to a row-wise list of dictionaries """

    if isinstance(x, dict):
        pass
    if isinstance(x, pd.DataFrame):
        x = x.to_dict(orient='records')

    # Drop None values - these do not need te be send each time
    if drop_na:
        x = [{k: v for k, v in x.items() if not pd.isna(v)} for x in x]

    return x


if __name__ == '__main__':

    pass
