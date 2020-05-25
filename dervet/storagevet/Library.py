"""
Library.py

Library of helper functions used in StorageVET.
"""

__author__ = 'Halley Nathwani, Micah Botkin-Levy, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import numpy as np
import pandas as pd
import copy

BOUND_NAMES = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']
# bound_names = ['ch_max', 'ch_min', 'dis_max', 'dis_min', 'ene_max', 'ene_min']
fr_obj_names = ['regu_d_cap', 'regu_c_cap', 'regd_d_cap', 'regd_c_cap', 'regu_d_ene', 'regu_c_ene', 'regd_d_ene', 'regd_c_ene']


def update_df(df1, df2):
    """ Helper function: Updates elements of df1 based on df2. Will add new columns if not in df1 or insert elements at
    the corresponding index if existing column

    Args:
        df1 (Data Frame): original data frame to be editted
        df2 (Data Frame): data frame to be added

    Returns:
        df1 (Data Frame)
    """

    old_col = set(df2.columns).intersection(set(df1.columns))
    df1 = df1.join(df2[list(set(df2.columns).difference(old_col))], how='left')  # join new columns
    df1.update(df2[list(old_col)])  # update old columns
    return df1


def disagg_col(df, group, col):
    """ Helper function: Adds a disaggregated column of 'col' based on the count of group
    TEMP FUNCTION: assumes that column is merged into disagg dataframe at original resolution = Bad approach

    Args:
        df (Data Frame): original data frame to be
        group (list): columns to group on
        col (string): column to disagg

    Returns:
        df (Data Frame)


    """
    # TODO this is a temp helper function until a more robust disagg function is built

    count_df = df.groupby(by=group).size()
    count_df.name = 'counts'
    df = df.reset_index().merge(count_df.reset_index(), on=group, how='left').set_index(df.index.names)
    df[col+'_disagg'] = df[col] / df['counts']
    return df


def create_opt_agg(df, n, dt):
    """ Helper function: Add opt_agg column to df based on n

    Args:
        df (Data Frame)
        n (): optimization control length
        dt (float): time step

    Returns:
        df (Data Frame)
    """
    # optimization level
    prev = 0
    df['opt_agg'] = 0
    # opt_agg period should not overlap multiple years
    for yr in df.index.year.unique():
        sub = copy.deepcopy(df[df.index.year == yr])
        if n == 'year':
            df.loc[df.index.year == yr, 'opt_agg'] = prev + 1  # continue counting from previous year opt_agg
        elif n == 'month':
            df.loc[df.index.year == yr, 'opt_agg'] = prev + sub.index.month  # continue counting from previous year opt_agg
        elif n == 'mpc':
            n = 1
            sub = copy.deepcopy(df[df.index.year == yr])
            sub['ind'] = range(len(sub))
            ind = (sub.ind // (n/dt)).astype(int) + 1
            df.loc[df.index.year == yr, 'opt_agg'] = ind + prev  # continue counting from previous year opt_agg
        else:  # assume n number of hours
            n = int(n)
            sub = copy.deepcopy(df[df.index.year == yr])
            sub['ind'] = range(len(sub))
            ind = (sub.ind // (n/dt)).astype(int)+1  # split year into groups of n days
            df.loc[df.index.year == yr, 'opt_agg'] = ind + prev  # continue counting from previous year opt_agg
        prev = max(df.opt_agg)

    return df


def apply_growth(source, rate, source_year, yr, freq):
    """ Applies linear growth rate to determine data for future year

    Args:
        source (Series): given data
        rate (float): yearly growth rate (%)
        source_year (Period): given data year
        yr (Period): future year to get data for
        freq (str): simulation time step frequency

    Returns:
        new (Series)
    """
    years = yr.year - source_year.year  # difference in years between source and desired yea
    new = source*(1+rate/100)**years  # apply growth rate to source data
    # new.index = new.index + pd.DateOffset(years=1)
    # deal with leap years
    source_leap = is_leap_yr(source_year.year)
    new_leap = is_leap_yr(yr.year)

    if (not source_leap) and new_leap:   # need to add leap day
        # if source is not leap year but desired year is, copy data from previous day
        new.index = new.index + pd.DateOffset(years=years)
        leap_ind = pd.date_range(start='02/29/'+str(yr), end='03/01/'+str(yr), freq=freq, closed='left')
        leap = pd.Series(new[leap_ind - pd.DateOffset(days=1)].values, index=leap_ind, name=new.name)
        new = pd.concat([new, leap])
        new = new.sort_index()
    elif source_leap and (not new_leap):  # need to remove leap day
        leap_ind = pd.date_range(start='02/29/'+str(source_year), end='03/01/'+str(source_year), freq=freq, closed='left')
        new = new[~new.index.isin(leap_ind)]
        new.index = new.index + pd.DateOffset(years=years)
    else:
        new.index = new.index + pd.DateOffset(years=years)
    return new


def is_leap_yr(year):
    """ Determines whether given year is leap year or not.

    Args:
        year (int): The year in question.

    Returns:
        bool: True for it being a leap year, False if not leap year.
    """
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0
