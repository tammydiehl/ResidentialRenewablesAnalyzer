"""
Scenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""

__author__ = 'Halley Nathwani, Thien Nyguen, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from ValueStreams.DAEnergyTimeShift import DAEnergyTimeShift
from ValueStreams.FrequencyRegulation import FrequencyRegulation
from ValueStreams.NonspinningReserve import NonspinningReserve
from ValueStreams.DemandChargeReduction import DemandChargeReduction
from ValueStreams.EnergyTimeShift import EnergyTimeShift
from ValueStreams.SpinningReserve import SpinningReserve
from ValueStreams.Backup import Backup
from ValueStreams.Deferral import Deferral
from ValueStreams.DemandResponse import DemandResponse
from ValueStreams.ResourceAdequacy import ResourceAdequacy
from ValueStreams.UserConstraints import UserConstraints
from ValueStreams.VoltVar import VoltVar
from ValueStreams.LoadFollowing import LoadFollowing
from Technology.BatteryTech import BatteryTech
from Technology.CAESTech import CAESTech
from Technology.CurtailPV import CurtailPV
from Technology.ICE import ICE
import Constraint as Const
import numpy as np
import pandas as pd
import Finances as Fin
import cvxpy as cvx
import Library as Lib
from prettytable import PrettyTable
import time
import sys
import copy
import logging

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Scenario(object):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario.

        Args:
            input_tree (Params.Params): Params of input attributes such as time_series, params, and monthly_data

        """
        self.deferral_df = None  # Initialized to none -- if active, this will not be None!

        self.verbose = input_tree.Scenario['verbose']
        u_logger.info("Creating Scenario...")

        self.active_objects = {
            'value streams': [],
            'distributed energy resources': [],
        }

        self.start_time = time.time()
        self.start_time_frmt = time.strftime('%Y%m%d%H%M%S')
        self.end_time = 0

        # add general case params (USER INPUTS)
        self.dt = input_tree.Scenario['dt']
        self.verbose_opt = input_tree.Scenario['verbose_opt']
        self.n = input_tree.Scenario['n']
        # self.n_control = input_tree.Scenario['n_control']
        self.n_control = 0
        self.mpc = input_tree.Scenario['mpc']

        self.start_year = input_tree.Scenario['start_year']
        self.end_year = input_tree.Scenario['end_year']
        self.opt_years = input_tree.Scenario['opt_years']
        self.incl_site_load = input_tree.Scenario['incl_site_load']
        self.incl_binary = input_tree.Scenario['binary']
        self.incl_slack = input_tree.Scenario['slack']
        # self.growth_rates = input_tree.Scenario['power_growth_rates']
        self.growth_rates = {'default': input_tree.Scenario['def_growth']}  # TODO: eventually turn into attribute
        self.frequency = input_tree.Scenario['frequency']

        self.no_export = input_tree.Scenario['no_export']
        self.no_import = input_tree.Scenario['no_import']

        self.customer_sided = input_tree.Scenario['customer_sided']

        self.technology_inputs_map = {
            'CAES': input_tree.CAES,
            'Battery': input_tree.Battery,
            'PV': input_tree.PV,
            'ICE': input_tree.ICE
        }

        self.predispatch_service_inputs_map = {
            'Deferral': input_tree.Deferral,
            'DR': input_tree.DR,
            'RA': input_tree.RA,
            'Backup': input_tree.Backup,
            'Volt': input_tree.Volt,
            'User': input_tree.User
        }
        self.service_input_map = {
            'DA': input_tree.DA,
            'FR': input_tree.FR,
            'LF': input_tree.LF,
            'SR': input_tree.SR,
            'NSR': input_tree.NSR,
            'DCM': input_tree.DCM,
            'retailTimeShift': input_tree.retailTimeShift,
        }

        self.solvers = set()

        # internal attributes to Case
        self.services = {}
        self.predispatch_services = {}
        self.technologies = {}
        self.absolute_constraints = {}

        # setup outputs
        self.results = pd.DataFrame()

        self.power_kw = self.prep_opt_results(input_tree.Scenario['power_timeseries'])
        self.aggregate_loads()
        self.objective_values = pd.DataFrame(index=np.sort(self.power_kw['opt_agg'].unique()))

        u_logger.info("Scenario Created Successfully...")

        # this is a flag used internally only to test the separation of absolute constraints from physical battery constraints
        # CHANGE IN STORAGE.PY -- the two values should match -- PLEASE REMOVE BEFORE RELEASING
        self.separate_constraints = False

    def prep_opt_results(self, time_series):
        """
        Create standard dataframe of power data from user time_series inputs

         Args:
            time_series (DataFrame): user time series data read from CSV
        Return:
            inputs_df (DataFrame): the required timeseries columns from the given time_series dataframe

        """

        inputs_df = time_series

        # calculate data for simulation of future years using growth rate
        inputs_df = self.add_growth_data(inputs_df, self.opt_years, self.verbose)
        # for service in self.services

        # create opt_agg (has to happen after adding future years)
        if self.mpc:
            inputs_df = Lib.create_opt_agg(inputs_df, 'mpc', self.dt)
        else:
            inputs_df = Lib.create_opt_agg(inputs_df, self.n, self.dt)

        u_logger.info("Finished preparing optimization results.")
        return inputs_df

    def aggregate_loads(self):
        """Add individual generation data to general generation columns, calculated from the inputted data.

        Notes:
            It doesnt make sense to aggrate all the different loads into one collective load

        """
        # logic to exclude site or aux load from load used
        self.power_kw['load'] = 0  # this is the aggregated load under the POI
        if self.incl_site_load:
            self.power_kw['load'] += self.power_kw['Site Load (kW)']

        self.power_kw = self.power_kw.sort_index()

    def add_growth_data(self, df, opt_years, verbose=False):
        """ Helper function: Adds rows to df where missing opt_years

        Args:
            df (DataFrame): given data
            opt_years (List): List of Period years where we need data for
            verbose (bool):

        Returns:
            df (DataFrame):

        TODO:
            might be a good idea to move back to Library
            change this to work with OOP framework
        """

        data_year = df.index.year.unique()  # which years was data given for
        # which years is data given for that is not needed
        dont_need_year = {pd.Period(year) for year in data_year} - {pd.Period(year) for year in opt_years}
        if len(dont_need_year) > 0:
            for yr in dont_need_year:
                df_sub = df[df.index.year != yr.year]  # choose all data that is not in the unneeded year
                df = df_sub

        data_year = df.index.year.unique()
        # which years do we not have data for
        no_data_year = {pd.Period(year) for year in opt_years} - {pd.Period(year) for year in data_year}
        # if there is a year we dont have data for
        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))  # which year to to apply growth rate to (is this the logic we want??)

                # create new dataframe for missing year
                new_index = pd.date_range(start='01/01/' + str(yr), end='01/01/' + str(yr + 1), freq=self.frequency, closed='left')
                new_data = pd.DataFrame(index=new_index)

                source_data = df[df.index.year == source_year.year]  # use source year data

                def_rate = self.growth_rates['default']

                # for each column in growth column
                for col in df.columns:
                    # look for specific growth rate in params, else use default growth rate
                    name = col.split(sep=' ')[0].lower()
                    col_type = col.split(sep=' ')[1].lower()
                    if col_type == 'load (kw)' or col_type == 'gen (kw/rated kw)':
                        # if name in self.growth_rates.keys():
                        #     rate = self.growth_rates[name]
                        # else:
                        u_logger.info('Using default growth rate (' + str(def_rate) + ') for' + str(name))
                        rate = def_rate
                    else:
                        rate = 0
                    new_data[col] = Lib.apply_growth(source_data[col], rate, source_year, yr, self.frequency)  # apply growth rate to column

                # add new year to original data frame
                df = pd.concat([df, new_data], sort=True)

        return df

    def init_financials(self, finance_inputs):
        """ Initializes the financial class with a copy of all the price data from timeseries, the tariff data, and any
         system variables required for post optimization analysis.

         Args:
             finance_inputs (Dict): Financial inputs

        """

        self.financials = Fin.Financial(finance_inputs)
        u_logger.info("Finished adding Financials...")

    def add_technology(self):
        """ Reads params and adds technology. Each technology gets initialized and their physical constraints are found.

        """
        ess_action_map = {
            'Battery': BatteryTech,
            'CAES': CAESTech
        }

        for storage in ess_action_map.keys():  # this will cause merging errors -HN
            inputs = self.technology_inputs_map[storage]
            if inputs is not None:
                tech_func = ess_action_map[storage]
                self.technologies['Storage'] = tech_func('Storage', self.power_kw['opt_agg'], inputs)
                u_logger.info("Finished adding storage...")

        generator_action_map = {
            'PV': CurtailPV,
            'ICE': ICE
        }

        for gen in generator_action_map.keys():
            inputs = self.technology_inputs_map[gen]
            if inputs is not None:
                tech_func = generator_action_map[gen]
                new_gen = tech_func(gen, inputs)
                new_gen.estimate_year_data(self.opt_years, self.frequency)
                self.technologies[gen] = new_gen
        u_logger.info("Finished adding generators...")
        self.active_objects['distributed energy resources'] = [self.technologies.keys()]
        u_logger.info("Finished adding active Technologies...")

    def add_services(self):
        """ Reads through params to determine which services are turned on or off. Then creates the corresponding
        service object and adds it to the list of services. Also generates a list of growth functions that apply to each
        service's timeseries data (to be used when adding growth data).

        Notes:
            This method needs to be applied after the technology has been initialized.
            ALL SERVICES ARE CONNECTED TO THE TECH

        """
        storage_inputs = self.technologies['Storage']

        predispatch_service_action_map = {
            'Deferral': Deferral,
            'DR': DemandResponse,
            'RA': ResourceAdequacy,
            'Backup': Backup,
            'Volt': VoltVar,
            'User': UserConstraints
        }
        for service, value in self.predispatch_service_inputs_map.items():
            if value is not None:
                u_logger.info("Using: " + str(service))
                inputs = self.predispatch_service_inputs_map[service]
                service_func = predispatch_service_action_map[service]
                new_service = service_func(inputs, self.technologies, self.power_kw, self.dt)
                new_service.estimate_year_data(self.opt_years, self.frequency)
                self.predispatch_services[service] = new_service

        u_logger.info("Finished adding Predispatch Services for Value Stream")

        service_action_map = {
            'DA': DAEnergyTimeShift,
            'FR': FrequencyRegulation,
            'SR': SpinningReserve,
            'NSR': NonspinningReserve,
            'DCM': DemandChargeReduction,
            'retailTimeShift': EnergyTimeShift,
            'LF': LoadFollowing
        }

        for service, value in self.service_input_map.items():
            if value is not None:
                u_logger.info("Using: " + str(service))
                inputs = self.service_input_map[service]
                service_func = service_action_map[service]
                new_service = service_func(inputs, storage_inputs, self.dt)
                new_service.estimate_year_data(self.opt_years, self.frequency)
                self.services[service] = new_service

        self.active_objects['value streams'] = [*self.predispatch_services.keys()] + [*self.services.keys()]
        u_logger.info("Finished adding Services for Value Stream")

        # only execute check_for_deferral_failure() in self.add_services() here I.F.F Deferral is not the only option
        if 'Deferral' in self.active_objects['value streams']:
            self.check_for_deferral_failure()

    def check_for_deferral_failure(self):
        # TODO: put this method in the Deferral class
        """ This functions checks the constraints of the storage system against any predispatch or user inputted constraints
        for any infeasible constraints on the system.

        The goal of this function is to predict the year that storage will fail to deferral a T&D asset upgrade.

        Only runs if Deferral is active.
        """
        u_logger.info('Finding first year of deferral failure...')
        deferral = self.predispatch_services['Deferral']
        tech = self.technologies['Storage']
        rte = tech.rte
        max_ch = tech.ch_max_rated
        max_dis = tech.dis_max_rated
        max_ene = tech.ene_max_rated * tech.ulsoc
        current_year = self.power_kw.index.year[-1]

        additional_years = [current_year]
        already_failed = False

        years_deferral_column = []
        min_power_deferral_column = []
        min_energy_deferral_column = []

        while current_year <= self.end_year.year:
            size = len(self.power_kw.index)
            print('current year: ' + str(current_year)) if self.verbose else None
            years_deferral_column.append(current_year)

            generation = np.zeros(size)
            if 'ICE' in self.active_objects['distributed energy resources']:
                # diesel generation is constant over the year, so add it out side the loop
                diesel = self.technologies['ICE']
                generation += np.repeat(diesel.rated_power*diesel.n, size)
            if 'PV' in self.active_objects['distributed energy resources']:
                generation += self.technologies['PV'].generation.values

            load = self.power_kw['load']
            min_power, min_energy = deferral.precheck_failure(self.dt, rte, load, generation)
            print(f'In {current_year} -- min power: {min_power}  min energy: {min_energy }') if self.verbose else None
            min_power_deferral_column.append(min_power)
            min_energy_deferral_column.append(min_energy)

            if (min_power > max_ch or min_power > max_dis or min_energy > max_ene) and not already_failed:
                # then we predict that deferral will fail
                last_deferral_yr = current_year - 1
                deferral.set_last_deferral_year(last_deferral_yr, current_year)

                new_opt_years = list(set(self.opt_years + additional_years))
                self.opt_years = new_opt_years
                already_failed = True

            # the current year we have could be the last year the deferral is possible, so we want
            # to keep it in self.opt_results until we know the next is can be deferred as well
            additional_years = [current_year, current_year + 1]

            # index the current year by one
            current_year += 1

            # remove columns that were not included in original timeseries (only pv and loads left)
            original_timeseries_headers = set(self.power_kw) - {'opt_agg', 'load'}
            power_kw_temp = self.power_kw.loc[:, original_timeseries_headers]

            new_opt_years = list(set(self.opt_years + additional_years))

            # add additional year of data to loads profiles
            self.power_kw = self.add_growth_data(power_kw_temp, new_opt_years, False)
            self.aggregate_loads()  # calculate the total net load

            # add additional year of PV generation forecast
            if 'PV' in self.technologies.keys():
                self.technologies['PV'].estimate_year_data(new_opt_years, self.frequency)

            # add additional year of data to deferred load
            deferral.estimate_year_data(new_opt_years, self.frequency)

        # add missing years of data to each value stream
        for service in self.services.values():
            service.estimate_year_data(self.opt_years, self.frequency)

        for service in self.predispatch_services.values():
            service.estimate_year_data(self.opt_years, self.frequency)

        # remove any data that we will not use for analysis
        if 'PV' in self.technologies.keys():
            self.technologies['PV'].estimate_year_data(self.opt_years, self.frequency)

        # print years that optimization will run for
        opt_years_str = '[  '
        for year in self.opt_years:
            opt_years_str += str(year) + '  '
        opt_years_str += ']'
        u_logger.info('Running analysis on years: ' + opt_years_str)

        # keep only the data for the years that the optimization will run on
        temp_df = pd.DataFrame()
        for year in self.opt_years:
            temp_df = pd.concat([temp_df, self.power_kw[self.power_kw.index.year == year]])
        self.power_kw = temp_df

        # create opt_agg (has to happen after adding future years)

        deferral_dict = {'Year': years_deferral_column, 'Power Capacity Requirement (kW)': min_power_deferral_column,
                         'Energy Capacity Requirement (kWh)': min_energy_deferral_column}

        if self.mpc:
            self.power_kw = Lib.create_opt_agg(self.power_kw, 'mpc', self.dt)
        else:
            self.power_kw = Lib.create_opt_agg(self.power_kw, self.n, self.dt)

        deferral_df = pd.DataFrame(deferral_dict)
        deferral_df.set_index('Year', inplace=True)
        self.deferral_df = deferral_df

    def add_control_constraints(self, deferral_check=False):
        """ This function collects 'absolute' constraints from the active value streams. Absolute constraints are
        time series constraints that will override the energy storage’s physical constraints. Graphically, the
        constraints that result from this function will lay on top of the physical constraints of that ESS to determine
        the acceptable envelope of operation. Therefore resulting in tighter maximum and minimum power/energy constraints.

        We create one general exogenous constraint for charge min, charge max, discharge min, discharge max, energy min,
        and energy max.

        Args:
            deferral_check (bool): flag to return non feasible timestamps if running deferral feasbility analysis

        """
        tech = self.technologies['Storage']

        if self.separate_constraints:
            ###### THIS CODE IS UNDERCONSTRUCTION ############
            feasible_check = None
            physical_constraints = tech.physical_constraints

            # initialize dataframe with values from physical_constraints
            temp_constraints = tech.calculate_control_constraints(self.power_kw.index)

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
                            temp_constraints.loc[absolute_index, name] = np.maximum(absolute_const, current_const)

                            # if the minimum value needed is greater than the physical maximum, infeasible scenario
                            max_value = physical_constraints[const_name + '_max' + '_rated'].value
                            if (temp_constraints[name] > max_value).any():
                                feasible_check = temp_constraints[temp_constraints[name] > max_value].index

                        else:
                            # if maximum constraint, choose lower constraint value
                            min_value = physical_constraints[const_name + '_min' + '_rated'].value
                            temp_constraints.loc[absolute_index.values, name] = np.minimum(absolute_const, current_const)

                            if (const_name == 'ene') & (temp_constraints[name] < min_value).any():
                                # if the maximum energy needed is less than the physical minimum, infeasible scenario
                                feasible_check = temp_constraints[temp_constraints[name] < min_value].index
                            else:
                                # it is ok to floor at zero since negative power max values will be handled in power min
                                # i.e negative ch_max means dis_min should be positive and ch_max should be 0)
                                temp_constraints[name] = temp_constraints[name].clip(lower=0)

            # now that we have a new list of constraints, create Constraint objects and store as 'control constraint'
            self.absolute_constraints = {'ene_min': Const.Constraint('ene_min', '', temp_constraints['ene_min']),
                                         'ene_max': Const.Constraint('ene_max', '', temp_constraints['ene_max']),
                                         'ch_min': Const.Constraint('ch_min', '', temp_constraints['ch_min']),
                                         'ch_max': Const.Constraint('ch_max', '', temp_constraints['ch_max']),
                                         'dis_min': Const.Constraint('dis_min', '', temp_constraints['dis_min']),
                                         'dis_max': Const.Constraint('dis_max', '', temp_constraints['dis_max'])}
        else:
            for service in self.predispatch_services.values():
                tech.add_value_streams(service, predispatch=True)  # just storage
            feasible_check = tech.calculate_control_constraints(self.power_kw.index)  # should pass any user inputted constraints here

        if (feasible_check is not None) & (not deferral_check):
            # if not running deferral failure analysis and infeasible scenario then stop and tell user
            u_logger.error('Predispatch and Storage inputs results in infeasible scenario')
            e_logger.error('Predispatch and Storage inputs results in infeasible scenario while adding control constraints.')
            quit()
        elif deferral_check:
            # return failure dttm to deferral failure analysis
            u_logger.info('Returned feasible_check to deferral failure analysis.')
            e_logger.error('Returned feasible_check to deferral failure analysis.')
            return feasible_check
        else:
            u_logger.info("Control Constraints Successfully Created...")

    def optimize_problem_loop(self, alpha=1):
        """ This function selects on opt_agg of data in self.time_series and calls optimization_problem on it.

        Args:
            alpha (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        """
        # remove any data that we will not use for analysis
        for service in self.services.values():
            service.estimate_year_data(self.opt_years, self.frequency)
        for service in self.predispatch_services.values():
            service.estimate_year_data(self.opt_years, self.frequency)
        if 'PV' in self.technologies.keys():
            self.technologies['PV'].estimate_year_data(self.opt_years, self.frequency)

        if 'Deferral' in self.predispatch_services.keys() and not len(self.services.keys()):
            # if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
            # then do not optimize (skip the optimization loop)
            return
        e_logger.info("Preparing Optimization Problem...")
        u_logger.info("Preparing Optimization Problem...")

        # list of all optimization windows
        periods = pd.Series(copy.deepcopy(self.power_kw.opt_agg.unique()))
        periods.sort_values()

        # for mpc
        window_shift = 0

        for ind, opt_period in enumerate(periods):

            # used to select rows from time_series relevant to this optimization window
            if not self.mpc:
                mask = self.power_kw.loc[:, 'opt_agg'] == opt_period
                # mask_index =
            else:
                mask = self.power_kw['opt_agg'].between(1 + int(self.n_control) * window_shift, int(self.n) + int(self.n_control) * window_shift)
                window_shift += 1

            # apply past degradation
            storage = self.technologies['Storage']
            storage.apply_past_degredation(ind, self.power_kw, mask, opt_period, self.n)

            print(time.strftime('%H:%M:%S') + ": Running Optimization Problem for " + str(self.power_kw.loc[mask].index[0]) + "...") if self.verbose else None

            # run optimization and return optimal variable and objective costs
            results, objective_values = self.optimization_problem(mask, alpha)

            # Add past degrade rate with degrade from calculated period
            storage = self.technologies['Storage']
            storage.calc_degradation(opt_period, results.index[0], results.index[-1], results['ene'])

            # add optimization variable results to power_kw
            if not results.empty and not self.mpc:
                self.power_kw = Lib.update_df(self.power_kw, results)
            elif not results.empty and self.mpc:
                results = results[:int(self.n_control)]
                self.power_kw = Lib.update_df(self.power_kw, results)

            # add objective expressions to financial obj_val
            if not objective_values.empty:
                objective_values.index = [opt_period]
                self.objective_values = Lib.update_df(self.objective_values, objective_values)

    def optimization_problem(self, mask, annuity_scalar=1):
        """ Sets up and runs optimization on a subset of data. Called within a loop.

        Args:
            mask (DataFrame): DataFrame of booleans used, the same length as self.time_series. The value is true if the
                        corresponding column in self.time_series is included in the data to be optimized.
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            variable_values (DataFrame): Optimal dispatch variables for each timestep in optimization period.

        """

        opt_var_size = int(np.sum(mask))

        # subset of input data relevant to this optimization period
        subs = self.power_kw.loc[mask, :]
        u_logger.info(subs.index.year[0])

        obj_const = []  # list of constraint costs (make this a dict for continuity??)
        variable_dic = {}  # Dict of optimization variables
        obj_expression = {}  # Dict of objective costs

        # default power and energy reservations (these could be filled in with optimization variables or costs below)
        power_reservations = np.array([0, 0, 0, 0])  # [c_max, c_min, d_max, d_min]
        energy_throughputs = [cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero'),
                              cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero'),
                              cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero')]  # [e_upper, e, e_lower]

        ##########################################################################
        # COLLECT OPTIMIZATION VARIABLES & POWER/ENERGY RESERVATIONS/THROUGHPUTS
        ########################################################################

        # add optimization variables for each technology
        # TODO [multi-tech] need to handle naming multiple optimization variables (i.e ch_1)
        for tech in self.technologies.values():
            variable_dic.update(tech.add_vars(opt_var_size))

        # calculate system generation
        generation = cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='gen_zero')
        if 'PV' in self.technologies.keys():
            generation += variable_dic['pv_out']
        if 'ICE' in self.technologies.keys():
            generation += variable_dic['ice_gen']

        value_streams = {**self.services, **self.predispatch_services}

        for stream in value_streams.values():
            # add optimization variables associated with each service
            variable_dic.update(stream.add_vars(opt_var_size))

            temp_power, temp_energy = stream.power_ene_reservations(variable_dic, mask)
            # add power reservations associated with each service
            power_reservations = power_reservations + np.array(temp_power)
            # add energy throughput and reservation associated with each service
            energy_throughputs = energy_throughputs + np.array(temp_energy)

        reservations = {'C_max': power_reservations[0],
                        'C_min': power_reservations[1],
                        'D_max': power_reservations[2],
                        'D_min': power_reservations[3],
                        'E': energy_throughputs[1],  # this represents the energy throughput of a value stream
                        'E_upper': energy_throughputs[0],  # max energy reservation (or throughput if called upon)
                        'E_lower': energy_throughputs[2]}  # min energy reservation (or throughput if called upon)

        #################################################
        # COLLECT OPTIMIZATION CONSTRAINTS & OBJECTIVES #
        #################################################

        # ADD IMPORT AND EXPORT CONSTRAINTS
        if self.no_export:
            obj_const += [cvx.NonPos(variable_dic['dis'] - variable_dic['ch'] + generation - subs.loc[:, 'load'])]
        if self.no_import:
            obj_const += [cvx.NonPos(-variable_dic['dis'] + variable_dic['ch'] - generation + subs.loc[:, 'load'])]

        pf_reliability = 'Reliability' in value_streams.keys() and len(value_streams.keys()) == 1 and value_streams['Reliability'].post_facto_only
        # add any constraints added by value streams
        for stream in value_streams.values():
            # add objective expression associated with each service
            obj_expression.update(stream.objective_function(variable_dic, subs, generation, annuity_scalar))

            obj_const += stream.objective_constraints(variable_dic, subs, generation, reservations)

        # add any objective costs from tech and the main physical constraints
        for tech in self.technologies.values():
            if not pf_reliability:
                obj_expression.update(tech.objective_function(variable_dic, mask, annuity_scalar))

            if self.mpc:
                try:
                    ene = self.power_kw['ene'].dropna().iloc[-1]
                except KeyError:
                    ene = None
                    e_logger.error('Key Error in energy reservation during optimization problem. Resulted in ene = None.')

                obj_const += tech.objective_constraints(variable_dic, mask, reservations, ene)
            else:
                obj_const += tech.objective_constraints(variable_dic, mask, reservations)

        obj = cvx.Minimize(sum(obj_expression.values()))
        prob = cvx.Problem(obj, obj_const)
        u_logger.info("Finished setting up the problem. Solving now.")

        try:  # TODO: better try catch statement --HN
            if prob.is_mixed_integer():
                # MBL: GLPK will solver to a default tolerance but can take a long time. Can use ECOS_BB which uses a branch and bound method
                # and input a tolerance but I have found that this is a more sub-optimal solution. Would like to test with Gurobi
                # information on solvers and parameters here: https://www.cvxpy.org/tgitstatutorial/advanced/index.html

                # prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB, mi_abs_eps=1, mi_rel_eps=1e-2, mi_max_iters=1000)
                start = time.time()
                prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                end = time.time()
                u_logger.info("Time it takes for solver to finish: " + str(end - start))
            else:
                start = time.time()
                # ECOS is default sovler and seems to work fine here
                prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB)
                end = time.time()
                u_logger.info("ecos solver")
                u_logger.info("Time it takes for solver to finish: " + str(end - start))
        except cvx.error.SolverError as err:
            e_logger.error("Solver Error. Exiting...")
            u_logger.error("Solver Error. Exiting...")
            sys.exit(err)

        #################################################
        # POST-OPTIMIZATION: COLLECT RESULTS TO RETURN #
        #################################################

        u_logger.info('Optimization problem was %s', str(prob.status))

        solution_found = True
        if prob.status == 'infeasible':
            # tell the user and throw an error specific to the problem being infeasible
            solution_found = False
            e_logger.error('Optimization problem was %s', str(prob.status))
            if self.verbose:
                print('Problem was INFEASIBLE. No solution found.')
            raise cvx.SolverError('Problem was infeasible. No solution found.')

        if prob.status == 'unbounded':
            solution_found = False
            # tell the user and throw an error specific to the problem being unbounded
            e_logger.error('Optimization problem was %s', str(prob.status))
            if self.verbose:
                print('Problem is UNBOUNDED. No solution found.')
            raise cvx.SolverError('Problem is unbounded. No solution found.')

        # save solver used
        self.solvers = self.solvers.union(prob.solver_stats.solver_name)

        cvx_types = (cvx.expressions.cvxtypes.expression(), cvx.expressions.cvxtypes.constant())
        # evaluate optimal objective expression
        obj_values = pd.DataFrame({name: [obj_expression[name].value if isinstance(obj_expression[name], cvx_types) else obj_expression[name]] for name in list(obj_expression)})
        # collect optimal dispatch variables
        for value in value_streams.values():
            value.save_variable_results(variable_dic, subs.index)
        for value in self.technologies.values():
            value.save_variable_results(variable_dic, subs.index)
        variable_values = pd.DataFrame({name: variable_dic[name].value for name in list(variable_dic)}, index=subs.index)

        if solution_found:
            # GENERAL CHECKS ON SOLUTION
            # check for non zero slack
            if np.any(abs(obj_values.filter(regex="_*slack$")) >= 1):
                u_logger.warning('WARNING! non-zero slack variables found in optimization solution')
                e_logger.warning('WARNING! non-zero slack variables found in optimization solution')

            # check for charging and discharging in same time step
            eps = 1e-4
            if any(((abs(variable_values['ch']) >= eps) & (abs(variable_values['dis']) >= eps)) & ('CAES' not in self.active_objects['distributed energy resources'])):
                u_logger.warning('WARNING! non-zero charge and discharge powers found in optimization solution. Try binary formulation')
                e_logger.warning('WARNING! non-zero charge and discharge powers found in optimization solution. Try binary formulation')

        # collect actual energy contributions from services
        for serv in self.services.values():
            if self.customer_sided:
                temp_ene_df = pd.DataFrame({'ene': np.zeros(len(subs.index))}, index=subs.index)
            else:
                sub_list = serv.e[-1].value.flatten('F')
                temp_ene_df = pd.DataFrame({'ene': sub_list}, index=subs.index)
            serv.ene_results = pd.concat([serv.ene_results, temp_ene_df], sort=True)
        return variable_values, obj_values

    @staticmethod
    def search_schema_type(root, attribute_name):
        """ Looks in the schema XML for the type of the attribute. Used to print the instance summary for previsualization.

        Args:
            root (object): the root of the input tree
            attribute_name (str): attribute being searched for

        Returns: the type of the attribute, if found. otherwise it returns "other"

        """
        for child in root:
            attributes = child.attrib
            if attributes.get('name') == attribute_name:
                if attributes.get('type') is None:
                    return "other"
                else:
                    return attributes.get('type')

    def instance_summary(self, input_tree):
        """ Prints each specific instance of this class, if there is sensitivity analysis, in the user log.

        Args:
            input_tree (dict): the input tree from Params.py

        Notes:
            Not used, but meant for sensitivity analysis

        """
        tree = input_tree.xmlTree
        treeRoot = tree.getroot()
        schema = input_tree.schema_tree

        u_logger.info("Printing summary table for each scenario...")
        table = PrettyTable()
        table.field_names = ["Category", "Element", "Active?", "Property", "Analysis?",
                             "Value", "Value Type", "Sensitivity"]
        for element in treeRoot:
            schemaType = self.search_schema_type(schema.getroot(), element.tag)
            activeness = element.attrib.get('active')
            for property in element:
                table.add_row([schemaType, element.tag, activeness, property.tag, property.attrib.get('analysis'),
                        property.find('Value').text, property.find('Type').text, property.find('Sensitivity').text])

        u_logger.info("\n" + str(table))
        u_logger.info("Successfully printed summary table for class Scenario in log file")
