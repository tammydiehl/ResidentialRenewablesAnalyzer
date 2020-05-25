"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'

import storagevet
import logging
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Constraint as Const
import copy
import re
import sys

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')
DEBUG = False


class BatterySizing(storagevet.BatteryTech):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, name,  opt_agg, params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            opt_agg (DataFrame): Initalized Financial Class
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        storagevet.BatteryTech.__init__(self, name,  opt_agg, params)

        self.user_duration = params['duration_max']

        self.size_constraints = []

        self.optimization_variables = {}

        # if the user inputted the energy rating as 0, then size for energy rating
        if not self.ene_max_rated:
            self.ene_max_rated = cvx.Variable(name='Energy_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ene_max_rated)]
            self.optimization_variables['ene_max_rated'] = self.ene_max_rated

        # if both the discharge and charge ratings are 0, then size for both and set them equal to each other
        if not self.ch_max_rated and not self.dis_max_rated:
            self.ch_max_rated = cvx.Variable(name='power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.dis_max_rated = self.ch_max_rated
            self.optimization_variables['ch_max_rated'] = self.ch_max_rated
            self.optimization_variables['dis_max_rated'] = self.dis_max_rated

        elif not self.ch_max_rated:  # if the user inputted the discharge rating as 0, then size discharge rating
            self.ch_max_rated = cvx.Variable(name='charge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.ch_max_rated)]
            self.optimization_variables['ch_max_rated'] = self.ch_max_rated

        elif not self.dis_max_rated:  # if the user inputted the charge rating as 0, then size for charge
            self.dis_max_rated = cvx.Variable(name='discharge_power_cap', integer=True)
            self.size_constraints += [cvx.NonPos(-self.dis_max_rated)]
            self.optimization_variables['dis_max_rated'] = self.dis_max_rated

        if self.user_duration:
            self.size_constraints += [cvx.NonPos((self.ene_max_rated / self.dis_max_rated) - self.user_duration)]

        self.capex = self.ccost + (self.ccost_kw * self.dis_max_rated) + (self.ccost_kwh * self.ene_max_rated)
        self.physical_constraints = {
            'ene_min_rated': Const.Constraint('ene_min_rated', self.name, self.llsoc * self.ene_max_rated),
            'ene_max_rated': Const.Constraint('ene_max_rated', self.name, self.ulsoc * self.ene_max_rated),
            'ch_min_rated': Const.Constraint('ch_min_rated', self.name, self.ch_min_rated),
            'ch_max_rated': Const.Constraint('ch_max_rated', self.name, self.ch_max_rated),
            'dis_min_rated': Const.Constraint('dis_min_rated', self.name, self.dis_min_rated),
            'dis_max_rated': Const.Constraint('dis_max_rated', self.name, self.dis_max_rated)}

    def calculate_duration(self):
        try:
            energy_rated = self.ene_max_rated.value
        except AttributeError:
            energy_rated = self.ene_max_rated

        try:
            dis_max_rated = self.dis_max_rated.value
        except AttributeError:
            dis_max_rated = self.dis_max_rated
        return energy_rated/dis_max_rated

    def objective_function(self, variables, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else alpha should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        storagevet.BatteryTech.objective_function(self, variables, mask, annuity_scalar)

        self.costs.update({'capex': self.capex})
        return self.costs

    def sizing_summary(self):
        """

        Returns: A datafram indexed by the terms that describe this DER's size and captial costs.

        """
        # obtain the size of the battery, these may or may not be optimization variable
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
        sizing_results = pd.DataFrame({'Energy Rating (kWh)': energy_rated,
                                       'Charge Rating (kW)': ch_max_rated,
                                       'Discharge Rating (kW)': dis_max_rated,
                                       'Round Trip Efficiency (%)': self.rte,
                                       'Lower Limit on SOC (%)': self.llsoc,
                                       'Upper Limit on SOC (%)': self.ulsoc,
                                       'Duration (hours)': energy_rated/dis_max_rated,
                                       'Capital Cost ($)': self.ccost,
                                       'Capital Cost ($/kW)': self.ccost_kw,
                                       'Capital Cost ($/kWh)': self.ccost_kwh}, index=index)
        if (sizing_results['Duration (hours)'] > 24).any():
            print('The duration of an Energy Storage System is greater than 24 hours!')
        return sizing_results

    def objective_constraints(self, variables, mask, reservations, mpc_ene=None):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []

        size = int(np.sum(mask))
        ene_target = self.soc_target * self.ene_max_rated

        # optimization variables
        ene = variables['ene']
        dis = variables['dis']
        ch = variables['ch']
        on_c = variables['on_c']
        on_d = variables['on_d']
        try:
            pv_gen = variables['pv_out']
        except KeyError:
            pv_gen = np.zeros(size)
        try:
            ice_gen = variables['ice_gen']
        except KeyError:
            ice_gen = np.zeros(size)

        if 'ene_max' in self.control_constraints.keys():
            ene_max = self.control_constraints['ene_max'].value[mask].values
            ene_max_t = ene_max[:-1]
            ene_max_n = ene_max[-1]
        else:
            ene_max = self.physical_constraints['ene_max_rated'].value
            ene_max_t = ene_max
            ene_max_n = ene_max

        if 'ene_min' in self.control_constraints.keys():
            ene_min = self.control_constraints['ene_min'].value[mask].values
            ene_min_t = ene_min[1:]
            ene_min_n = ene_min[-1]
        else:
            ene_min = self.physical_constraints['ene_min_rated'].value
            ene_min_t = ene_min
            ene_min_n = ene_min

        if 'ch_max' in self.control_constraints.keys():
            ch_max = self.control_constraints['ch_max'].value[mask].values
        else:
            ch_max = self.physical_constraints['ch_max_rated'].value

        if 'ch_min' in self.control_constraints.keys():
            ch_min = self.control_constraints['ch_min'].value[mask].values
        else:
            ch_min = self.physical_constraints['ch_min_rated'].value

        if 'dis_max' in self.control_constraints.keys():
            dis_max = self.control_constraints['dis_max'].value[mask].values
        else:
            dis_max = self.physical_constraints['dis_max_rated'].value

        if 'dis_min' in self.control_constraints.keys():
            dis_min = self.control_constraints['dis_min'].value[mask].values
        else:
            dis_min = self.physical_constraints['dis_min_rated'].value

        # energy at the end of the last time step
        constraint_list += [cvx.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - reservations['E'][-1] + (self.dt * ene[-1] * self.sdr * 0.01))]

        # energy generally for every time step
        constraint_list += [cvx.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - reservations['E'][:-1] + (self.dt * ene[:-1] * self.sdr * 0.01))]

        # energy at the beginning of the optimization window
        if mpc_ene is None:
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
        else:
            constraint_list += [cvx.Zero(ene[0] - mpc_ene)]

        # Keep energy in bounds determined in the constraints configuration function
        constraint_list += [cvx.NonPos(ene_target - ene_max_n + reservations['E_upper'][-1] - variables['ene_max_slack'][-1])]  # TODO: comment out if putting energy user constraint and infeasible result
        constraint_list += [cvx.NonPos(ene[:-1] - ene_max_t + reservations['E_upper'][:-1] - variables['ene_max_slack'][:-1])]

        constraint_list += [cvx.NonPos(-ene_target + ene_min_n - (pv_gen[-1]*self.dt) - (ice_gen[-1]*self.dt) - reservations['E_lower'][-1] - variables['ene_min_slack'][-1])]
        constraint_list += [cvx.NonPos(ene_min_t - (pv_gen[1:]*self.dt) - (ice_gen[1:]*self.dt) - ene[1:] + reservations['E_lower'][:-1] - variables['ene_min_slack'][:-1])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(ch - cvx.multiply(ch_max, on_c) - variables['ch_max_slack'])]
        constraint_list += [cvx.NonPos(ch - ch_max + reservations['C_max'] - variables['ch_max_slack'])]

        constraint_list += [cvx.NonPos(cvx.multiply(ch_min, on_c) - ch - variables['ch_min_slack'])]
        constraint_list += [cvx.NonPos(ch_min - ch + reservations['C_min'] - variables['ch_min_slack'])]

        constraint_list += [cvx.NonPos(dis - cvx.multiply(dis_max, on_d) - variables['dis_max_slack'])]
        constraint_list += [cvx.NonPos(dis - dis_max + reservations['D_max'] - variables['dis_max_slack'])]

        constraint_list += [cvx.NonPos(cvx.multiply(dis_min, on_d) - dis - variables['dis_min_slack'])]
        constraint_list += [cvx.NonPos(dis_min - dis + reservations['D_min'] - variables['dis_min_slack'])]
        # constraints to keep slack variables positive
        if self.incl_slack:
            constraint_list += [cvx.NonPos(-variables['ch_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ch_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_min_slack'])]

        if self.incl_binary:
            # when dis_min or ch_min has been overwritten (read: increased) by predispatch services, need to force technology to be on
            # TODO better way to do this???
            if 'dis_min' in self.control_constraints.keys():
                ind_d = [i for i in range(size) if self.control_constraints['dis_min'].value[mask].values[i] > self.physical_constraints['dis_min_rated'].value]
                if len(ind_d) > 0:
                    constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
            if 'ch_min' in self.control_constraints.keys():
                ind_c = [i for i in range(size) if self.control_constraints['ch_min'].value[mask].values[i] > self.physical_constraints['ch_min_rated'].value]
                if len(ind_c) > 0:
                    constraint_list += [on_c[ind_c] == 1]  # np.ones(len(ind_c))

            # note: cannot operate startup without binary
            if self.incl_startup:
                # startup variables are positive
                constraint_list += [cvx.NonPos(-variables['start_d'])]
                constraint_list += [cvx.NonPos(-variables['start_c'])]
                # difference between binary variables determine if started up in previous interval
                constraint_list += [cvx.NonPos(cvx.diff(on_d) - variables['start_d'][1:])]  # first variable not constrained
                constraint_list += [cvx.NonPos(cvx.diff(on_c) - variables['start_c'][1:])]  # first variable not constrained

        constraint_list += self.size_constraints

        return constraint_list

    def calculate_control_constraints(self, datetimes):
        """ Generates a list of master or 'control constraints' from physical constraints and all
        predispatch service constraints.

        Args:
            datetimes (list): The values of the datetime column within the initial time_series data frame.

        Returns:
            Array of datetimes where the control constraints conflict and are infeasible. If all feasible return None.

        Note: the returned failed array returns the first infeasibility found, not all feasibilities.
        TODO: come back and check the user inputted constraints --HN
        """
        # create temp dataframe with values from physical_constraints
        temp_constraints = pd.DataFrame(index=datetimes)

        # create a df with all physical constraint values
        for constraint in self.physical_constraints.values():
            temp_constraints[re.search('^.+_.+_', constraint.name).group(0)[0:-1]] = copy.deepcopy(constraint.value)

        # change physical constraint with predispatch service constraints at each timestep
        # predispatch service constraints should be absolute constraints
        for service in self.predispatch_services.values():
            for constraint in service.constraints.values():
                if constraint.value is not None:
                    strp = constraint.name.split('_')
                    const_name = strp[0]
                    const_type = strp[1]
                    name = const_name + '_' + const_type
                    absolute_const = constraint.value.values  # constraint values
                    absolute_index = constraint.value.index  # the datetimes for which the constraint applies

                    current_const = temp_constraints.loc[absolute_index, name].values  # value of the current constraint

                    if const_type == "min":
                        # if minimum constraint, choose higher constraint value
                        try:
                            temp_constraints.loc[absolute_index, name] = np.maximum(absolute_const, current_const)
                        except (TypeError, SystemError):
                            temp_constraints.loc[absolute_index, name] = absolute_const
                        # temp_constraints.loc[constraint.value.index, name] += constraint.value.values

                        # if the minimum value needed is greater than the physical maximum, infeasible scenario
                        max_value = self.physical_constraints[const_name + '_max' + '_rated'].value
                        try:
                            constraint_violation = any(temp_constraints[name] > max_value)
                        except (ValueError, TypeError, SystemError):
                            constraint_violation = False
                        if constraint_violation:
                            return temp_constraints[temp_constraints[name] > max_value].index

                    else:
                        # if maximum constraint, choose lower constraint value
                        try:
                            temp_constraints.loc[absolute_index, name] = np.minimum(absolute_const, current_const)
                        except (TypeError, SystemError):
                            temp_constraints.loc[absolute_index, name] = absolute_const
                        # temp_constraints.loc[constraint.value.index, name] -= constraint.value.values

                        # if the maximum energy needed is less than the physical minimum, infeasible scenario
                        min_value = self.physical_constraints[const_name + '_min' + '_rated'].value
                        try:
                            constraint_violation = any(temp_constraints[name] < min_value)
                        except (ValueError, TypeError):
                            constraint_violation = False
                        if (const_name == 'ene') & constraint_violation:

                            return temp_constraints[temp_constraints[name] > max_value].index
                        else:
                            # it is ok to floor at zero since negative power max values will be handled in power min
                            # i.e negative ch_max means dis_min should be positive and ch_max should be 0)
                            temp_constraints[name] = temp_constraints[name].clip(lower=0)
                    self.control_constraints.update({constraint.name: Const.Constraint(constraint.name, self.name, temp_constraints[constraint.name])})

        # # now that we have a new list of constraints, create Constraint objects and store as 'control constraint'
        # self.control_constraints = {'ene_min': Const.Constraint('ene_min', self.name, temp_constraints['ene_min']),
        #                             'ene_max': Const.Constraint('ene_max', self.name, temp_constraints['ene_max']),
        #                             'ch_min': Const.Constraint('ch_min', self.name, temp_constraints['ch_min']),
        #                             'ch_max': Const.Constraint('ch_max', self.name, temp_constraints['ch_max']),
        #                             'dis_min': Const.Constraint('dis_min', self.name, temp_constraints['dis_min']),
        #                             'dis_max': Const.Constraint('dis_max', self.name, temp_constraints['dis_max'])}
        return None

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        # recacluate capex before reporting proforma
        self.capex = self.ccost + (self.ccost_kw * self.dis_max_rated) + (self.ccost_kwh * self.ene_max_rated)
        proforma = super().proforma_report(opt_years, results)
        return proforma

    # def physical_properties(self):
    #     """
    #
    #     Returns: a dictionary of physical properties that define the ess
    #         includes 'charge max', 'discharge max, 'operation soc min', 'operation soc max', 'rte', 'energy cap'
    #
    #     """
    #     try:
    #         energy_rated = self.ene_max_rated.value
    #     except AttributeError:
    #         energy_rated = self.ene_max_rated
    #
    #     try:
    #         ch_max_rated = self.ch_max_rated.value
    #     except AttributeError:
    #         ch_max_rated = self.ch_max_rated
    #
    #     try:
    #         dis_max_rated = self.dis_max_rated.value
    #     except AttributeError:
    #         dis_max_rated = self.dis_max_rated
    #
    #     ess_properties = {'charge max': ch_max_rated,
    #                       'discharge max': dis_max_rated,
    #                       'rte': self.rte,
    #                       'energy cap': energy_rated,
    #                       'operation soc min': self.llsoc,
    #                       'operation soc max': self.ulsoc}
    #     return ess_properties

    def being_sized(self):
        """ checks itself to see if this instance is being sized

        Returns: true if being sized, false if not being sized

        """
        return bool(len(self.size_constraints))
