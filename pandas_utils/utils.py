import pandas as pd
from typing import Tuple, List

__all__ = ['groupcount_differences_between_rows']


def groupcount_differences_between_rows(
        df1: pd.DataFrame,
        df2: pd.DataFrame
) -> pd.DataFrame:
    """
     Group differences between rows and count entries of each group.

    Example:
        Input:
            df1:            df2:
            A   B   C       A   B   C
            ----------      ----------
            1   1   1       1   1   1
            1   1   1       1   99  1
            1   1   1       1   99  1
            1   1   1       1   1   99
            1   1   1       99  99  1

        Output:
            group     count     indices
            ---------------------------
            [B]         2       [1, 2]
            [C]         1       [3]
            [B, C]      1       [4]

    For more examples see the tests of this function.
    """
    df1, df2 = check_and_parse_dataframes(df1, df2)
    diffs = pd.DataFrame(columns=["group", "count", "indices"])
    
    for (i, row1), (_, row2) in zip(df1.iterrows(), df2.iterrows()):
        group = get_names_of_unequal_fields(row1, row2)
        if group:
            diffs = diffs.append(dict(
                group=tuple(group),
                count=1,
                indices=[i]
            ), ignore_index=True)
    diffs = (diffs.groupby("group", as_index=False)
             .agg({'count': 'sum', 'indices': 'sum'}))
    return diffs


def get_names_of_unequal_fields(
        row1: pd.Series,
        row2: pd.Series
) -> List[str]:
    return [str(name) for (name, field1), (_, field2) in zip(row1.items(), row2.items()) if field1 != field2]


def check_and_parse_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame
) -> Tuple[pd.DataFrame]:
    if df1.shape != df2.shape:
        raise ValueError(f'df1 and df2 do not have same shape: {df1.shape}, {df2.shape}')
    for df in (df1, df2):
        if len(df.columns) != len(set(df.columns)):
            raise ValueError(f"Dataframes are not allowed to have non-unique columns: {df1.columns}, {df2.columns}")
    if set(df1.columns) != set(df2.columns):
        raise ValueError(f'df1 and df2 do not have the same set of columns: {df1.columns}, {df2.columns}')
    if any(list(df1.columns != df2.columns)):
        df2 = df2[df1.columns]
    if any(list(df1.dtypes != df2.dtypes)):
        raise TypeError(f'Datatypes of same named columns in df1 and df2 do not match:')
    return df1, df2


if __name__ == '__main__':
    df1 = pd.DataFrame(dict(
        A=[1, 1, 1],
        B=[1, 1, 1],
        C=[1, 1, 1]
    ))
    df2 = pd.DataFrame(dict(
        A=[1, 9, 9],
        B=[9, 1, 1],
        C=[9, 9, 9]
    ))
    observation = groupcount_differences_between_rows(df1, df2)
    
    diffs2 = pd.DataFrame(dict(
        group=[("Comments",), ("Comments",), ("Old_Price_EUR",), ("Old_Price_EUR",)],
        count=[1, 1, 1, 1],
        indices=[[0], [1], [5], [6]]
    ))
