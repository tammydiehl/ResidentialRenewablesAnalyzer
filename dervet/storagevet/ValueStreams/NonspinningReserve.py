"""
NonspinningReserve.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from .ValueStream import ValueStream
import numpy as np
import cvxpy as cvx
import pandas as pd
import logging
try:
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class NonspinningReserve(ValueStream):
    """ Nonspinning Reserve. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)
        """
        ValueStream.__init__(self, tech, 'NSR', dt)
        self.price = params['price']
        self.growth = params['growth']
        self.dis_power = params['dis_power']
        self.duration = params['duration']

        self.variable_names = {'nsr_c', 'nsr_d'}
        self.variables = pd.DataFrame(columns=self.variable_names)

    @staticmethod
    def add_vars(size):
        """ Adds optimization variables to dictionary

        Variables added:
            nsr_d (Variable): A cvxpy variable for non-spinning reserve capacity to increase discharging power
            nsr_c (Variable): A cvxpy variable for non-spinning reserve capacity to decrease charging power

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        return {'nsr_d': cvx.Variable(shape=size, name='nsr_d'),
                'nsr_c': cvx.Variable(shape=size, name='nsr_c')}

    def objective_function(self, variables, subs, generation, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            variables (Dict): dictionary of variables being optimized
            subs (DataFrame): table of load data for the optimization windows
            generation (list, Expression): the sum of generation within the system
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            The expression of the objective function that it affects. This can be passed into the cvxpy solver.

        """

        p_nsr = cvx.Parameter(subs.index.size, value=self.price.loc[subs.index].values, name='price')

        return {self.name: cvx.sum(-p_nsr*variables['nsr_c'] - p_nsr*variables['nsr_d']) * self.dt * annuity_scalar}

    def objective_constraints(self, variables, subs, generation, reservations=None):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            subs (DataFrame): Subset of time_series data that is being optimized
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
                reservations (Dict): power reservations from dispatch services

        Returns:

        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-variables['nsr_c'])]
        constraint_list += [cvx.NonPos(-variables['nsr_d'])]
        constraint_list += [cvx.NonPos(variables['nsr_d'] - self.dis_power)]
        return constraint_list

    def power_ene_reservations(self, opt_vars, mask):
        """ Determines power and energy reservations required at the end of each timestep for the service to be provided.
        Additionally keeps track of the reservations per optimization window so the values maybe accessed later.

        Args:
            opt_vars (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A power reservation and a energy reservation array for the optimization window--
            C_max, C_min, D_max, D_min, E_upper, E, and E_lower (in that order)
        """
        eta = self.storage.rte
        size = opt_vars['ene'].shape

        # calculate reservations
        c_max = 0
        c_min = opt_vars['nsr_c']
        d_min = 0
        d_max = opt_vars['nsr_d']
        e_upper = cvx.Parameter(shape=size, value=np.zeros(size), name='e_upper')
        e = cvx.Parameter(shape=size, value=np.zeros(size), name='e')
        e_lower = opt_vars['nsr_c']*eta*self.dt + opt_vars['nsr_d']*self.duration

        # save reservation for optimization window
        self.e.append(e)
        self.e_lower.append(e_lower)
        self.e_upper.append(e_upper)
        self.c_max.append(c_max)
        self.c_min.append(c_min)
        self.d_max.append(d_max)
        self.d_min.append(d_min)
        return [c_max, c_min, d_max, d_min], [e_upper, e, e_lower]

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        data_year = self.price.index.year.unique()
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                source_data = self.price[self.price.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.growth, source_year, yr, frequency)
                self.price = pd.concat([self.price, new_data], sort=True)  # add to existing

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, "NSR Price Signal ($/kW)"] = self.price
        report.loc[:, 'Non-spinning Reserve (Charging) (kW)'] = self.variables['nsr_c']
        report.loc[:, 'Non-spinning Reserve (Discharging) (kW)'] = self.variables['nsr_d']
        return report

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A tuple of a DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided), a list (of columns that remain zero), and a list (of columns that
        retain a constant value over the entire project horizon).

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        proforma, _, _ = ValueStream.proforma_report(self, opt_years, results)

        nonspin_bid = results.loc[:, 'Non-spinning Reserve (Charging) (kW)'] + results.loc[:, 'Non-spinning Reserve (Discharging) (kW)']
        nonspinning_prof = np.multiply(nonspin_bid, self.price) * self.dt

        for year in opt_years:
            year_subset = nonspinning_prof[nonspinning_prof.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), 'Non-Spinning Reserves'] = year_subset.sum()

        return proforma, None, None

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = time_series_data.loc[:, 'NSR Price ($/kW)']
        except KeyError:
            pass
