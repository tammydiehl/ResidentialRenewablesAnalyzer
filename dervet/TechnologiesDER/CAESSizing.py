"""
CAESSizing.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Thien Nguyen'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Thien Nguyen', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'

import storagevet
import cvxpy as cvx
import logging
import pandas as pd
import numpy as np
import storagevet.Constraint as Const

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CAESSizing(storagevet.CAESTech):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, name,  opt_agg, params):
        """ Initializes CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (DataFrame): Initialized Financial Class
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        storagevet.CAESTech.__init__(self, name,  opt_agg, params)

        self.size_constraints = []

        self.optimization_variables = {}

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            caes_ene (Variable): A cvxpy variable for Energy at the end of the time step
            caes_dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step
            caes_ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step
            caes_ene_max_slack (Variable): A cvxpy variable for energy max slack
            caes_ene_min_slack (Variable): A cvxpy variable for energy min slack
            caes_ch_max_slack (Variable): A cvxpy variable for charging max slack
            caes_ch_min_slack (Variable): A cvxpy variable for charging min slack
            caes_dis_max_slack (Variable): A cvxpy variable for discharging max slack
            caes_dis_min_slack (Variable): A cvxpy variable for discharging min slack

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'caes_ene': cvx.Variable(shape=size, name='caes_ene'),
                     'caes_dis': cvx.Variable(shape=size, name='caes_dis'),
                     'caes_ch': cvx.Variable(shape=size, name='caes_ch'),
                     'caes_ene_max_slack': cvx.Parameter(shape=size, name='caes_ene_max_slack', value=np.zeros(size)),
                     'caes_ene_min_slack': cvx.Parameter(shape=size, name='caes_ene_min_slack', value=np.zeros(size)),
                     'caes_dis_max_slack': cvx.Parameter(shape=size, name='caes_dis_max_slack', value=np.zeros(size)),
                     'caes_dis_min_slack': cvx.Parameter(shape=size, name='caes_dis_min_slack', value=np.zeros(size)),
                     'caes_ch_max_slack': cvx.Parameter(shape=size, name='caes_ch_max_slack', value=np.zeros(size)),
                     'caes_ch_min_slack': cvx.Parameter(shape=size, name='caes_ch_min_slack', value=np.zeros(size)),
                     'caes_on_c': cvx.Parameter(shape=size, name='caes_on_c', value=np.ones(size)),
                     'caes_on_d': cvx.Parameter(shape=size, name='caes_on_d', value=np.ones(size)),
                     }

        if self.incl_slack:
            self.variable_names.update(['caes_ene_max_slack', 'caes_ene_min_slack', 'caes_dis_max_slack', 'caes_dis_min_slack', 'caes_ch_max_slack', 'caes_ch_min_slack'])
            variables.update({'caes_ene_max_slack': cvx.Variable(shape=size, name='caes_ene_max_slack'),
                              'caes_ene_min_slack': cvx.Variable(shape=size, name='caes_ene_min_slack'),
                              'caes_dis_max_slack': cvx.Variable(shape=size, name='caes_dis_max_slack'),
                              'caes_dis_min_slack': cvx.Variable(shape=size, name='caes_dis_min_slack'),
                              'caes_ch_max_slack': cvx.Variable(shape=size, name='caes_ch_max_slack'),
                              'caes_ch_min_slack': cvx.Variable(shape=size, name='caes_ch_min_slack')})
        if self.incl_binary:
            self.variable_names.update(['caes_on_c', 'caes_on_d'])
            variables.update({'caes_on_c': cvx.Variable(shape=size, boolean=True, name='caes_on_c'),
                              'caes_on_d': cvx.Variable(shape=size, boolean=True, name='caes_on_d')})
            if self.incl_startup:
                self.variable_names.update(['bat_start_c', 'bat_start_d'])
                variables.update({'caes_start_c': cvx.Variable(shape=size, name='caes_start_c'),
                                  'caes_start_d': cvx.Variable(shape=size, name='caes_start_d')})

        variables.update(self.optimization_variables)

        return variables

    def sizing_summary(self):
        """
        TODO: CAESSizing is waiting to be implemented, it is currently mimicking BatterySizing's method

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the CAES, these may or may not be optimization variable
        # therefore we check to see if it is by trying to get its value attribute in a try-except statement.
        # If there is an error, then we know that it was user inputted and we just take that value instead.
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        try:
            ch_max_rated = self.ch_max_rated.value
        except AttributeError:
            ch_max_rated = self.ch_max_rated

        try:
            dis_max_rated = self.dis_max_rated.value
        except AttributeError:
            dis_max_rated = self.dis_max_rated

        index = pd.Index([self.name], name='DER')
        sizing_results = pd.DataFrame({'CAES Energy Rating (kWh)': energy_rated,
                                       'CAES Charge Rating (kW)': ch_max_rated,
                                       'CAES Discharge Rating (kW)': dis_max_rated,
                                       'CAES Duration (hours)': energy_rated / dis_max_rated,
                                       'CAES Capital Cost ($)': self.ccost,
                                       'CAES Capital Cost ($/kW)': self.ccost_kw,
                                       'CAES Capital Cost ($/kWh)': self.ccost_kwh}, index=index)
        return sizing_results

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = storagevet.CAESTech.timeseries_report(self)
        results[self.name + ' CAES Discharge (kW)'] = self.variables['caes_dis']
        results[self.name + ' CAES Charge (kW)'] = self.variables['caes_ch']
        results[self.name + ' CAES Power (kW)'] = self.variables['caes_dis'] - self.variables['caes_ch']
        results[self.name + ' CAES State of Energy (kWh)'] = self.variables['caes_ene']

        try:
            energy_rate = self.ene_max_rated.value
        except AttributeError:
            energy_rate = self.ene_max_rated

        results['CAES SOC (%)'] = self.variables['caes_ene'] / energy_rate
        results['CAES Fuel Price ($)'] = self.fuel_price

        return results





