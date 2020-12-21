import pytest
import pandas as pd
from pandas_utils import groupcount_differences_between_rows


class TestGroupcountDifferencesBetweenRows:
    """
    Function is tested for the following row field datatypes:
        - int
        - float
        - str
        - list
        - tuple
    """

    def test_single_cause_differences_between_rows(self):
        """All row pairs that differ, differ because of a single entry."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1, 1, 1],
            B=[1, 1, 1],
            C=[1, 1, 1]
        ))
        df2 = pd.DataFrame(dict(
            A=[9, 1, 9],
            B=[1, 9, 1],
            C=[1, 1, 1]
        ))
        expectation = pd.DataFrame(dict(
            group=[("A",), ("B",)],
            count=[2, 1],
            indices=[[0, 2], [1]]
        ))
        # fmt: on
        observation = groupcount_differences_between_rows(df1, df2)
        assert observation.equals(expectation)

    def test_multi_cause_differences_between_rows(self):
        """All row pairs that differ, differ because of a multiple entries."""
        # fmt: off
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
        expectation = pd.DataFrame(dict(
            group=[("A", "C"), ("B", "C")],
            count=[2, 1],
            indices=[[1, 2], [0]]
        ))
        # fmt: on
        observation = groupcount_differences_between_rows(df1, df2)
        assert observation.equals(expectation)

    def test_different_datatypes_across_fields(self):
        """Different datatypes in same named columns are not allowed (tested in another test). However,
        different datatypes across fields should work."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1.1, 1.2, "string", "string", [1, 1], (1, 1)],
        ))
        df2 = pd.DataFrame(dict(
            A=[1.1, 9.8, "string", "string2", [1, 1], (1, 2)],
        ))
        expectation = pd.DataFrame(dict(
            group=[("A",)],
            count=[3],
            indices=[[1, 3, 5]]
        ))
        # fmt: on
        observation = groupcount_differences_between_rows(df1, df2)
        assert observation.equals(expectation)

    def test_equal_rows(self):
        """All row entries of all row pairs are identical."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1, 1, 1]
        ))
        df2 = pd.DataFrame(dict(
            A=[1, 1, 1]
        ))
        # fmt: on
        expectation = pd.DataFrame(columns=["group", "count", "indices"])
        observation = groupcount_differences_between_rows(df1, df2)
        assert have_same_column_names_and_are_empty(observation, expectation)

    def test_empty_dataframes(self):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        expectation = pd.DataFrame(columns=["group", "count", "indices"])
        observation = groupcount_differences_between_rows(df1, df2)
        assert have_same_column_names_and_are_empty(observation, expectation)

    def test_dataframes_with_different_column_orders(self):
        """
        Dataframes have identical set of columns, however column order differs between them. In this case both
        dataframes should be first ordered by the column order of df1 and then be compared to each other row-wise.
        """
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1, 1],
            B=[1, 1],
        ))
        df2 = pd.DataFrame(dict(
            B=[1, 1],
            A=[2, 1],
        ))
        expectation = pd.DataFrame(dict(
            group=[("A",)],
            count=[1],
            indices=[[0]]
        ))
        # fmt: on
        observation = groupcount_differences_between_rows(df1, df2)
        assert observation.equals(expectation)

    def test_raise_error_if_dataframes_have_different_lengths(self):
        """If dataframes do not have equal length a ValueError is expected to be raised."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1]
        ))
        df2 = pd.DataFrame(dict(
            A=[1, 1]
        ))
        # fmt: on
        with pytest.raises(ValueError):
            groupcount_differences_between_rows(df1, df2)

    def test_raise_error_if_dataframes_have_different_widths(self):
        """If dataframes do not have equal number of columns a ValueError is expected to be raised."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1]
        ))
        df2 = pd.DataFrame(dict(
            A=[1],
            B=[1]
        ))
        # fmt: on
        with pytest.raises(ValueError):
            groupcount_differences_between_rows(df1, df2)

    def test_raise_error_if_dataframes_have_non_unique_columns(self):
        """If dataframes do not have non-unique columns a ValueError is expected to be raised."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1],
            B=[1]
        ))
        df2 = pd.DataFrame(dict(
            A=[1],
            B=[1]
        ))
        # fmt: on
        df1, df2 = [df.rename(columns={"B": "A"}) for df in (df1, df2)]
        with pytest.raises(ValueError):
            groupcount_differences_between_rows(df1, df2)

    def test_raise_error_if_dataframes_have_different_column_sets(self):
        """If dataframes do not have equal column sets a ValueError is expected to be raised."""
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=[1],
        ))
        df2 = pd.DataFrame(dict(
            B=[1],
            A=[1],
        ))
        # fmt: on
        with pytest.raises(ValueError):
            groupcount_differences_between_rows(df1, df2)

    def test_raise_error_if_same_named_columns_have_different_datatypes(self):
        # fmt: off
        df1 = pd.DataFrame(dict(
            A=["string"],
        ))
        df2 = pd.DataFrame(dict(
            A=[1],
        ))
        # fmt: on
        with pytest.raises(TypeError):
            groupcount_differences_between_rows(df1, df2)


def have_same_column_names_and_are_empty(df1, df2):
    return all(df1.columns == df2.columns) & df1.empty & df2.empty
