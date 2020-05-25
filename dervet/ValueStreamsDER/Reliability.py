"""
Reliability.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Suma Jothibasu, Halley Nathwani and Miles Evans'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'

import storagevet.Constraint as Const
import numpy as np
import storagevet
import cvxpy as cvx
import pandas as pd
import time
import logging


u_logger = logging.getLogger('User')
DEBUG = False


class Reliability(storagevet.ValueStream):
    """ Reliability Service. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params, techs, load_data, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            load_data (DataFrame): table of time series load data
            dt (float): optimization timestep (hours)
        """

        # generate the generic predispatch service object
        super().__init__(None, 'Reliability', dt)
        self.outage_duration_coverage = params['target']  # must be in hours
        self.dt = params['dt']
        self.post_facto_only = params['post_facto_only']
        self.nu = params['nu'] / 100
        self.gamma = params['gamma'] / 100
        self.max_outage_duration = params['max_outage_duration']
        # self.n_2 = params['n-2']
        self.n_2 = 0
        self.contribution_df = pd.DataFrame()

        if 'Diesel' in techs.keys():
            self.ice_rated_power = techs['Diesel'].rated_power
        # else:
        #     self.ice_rated_power = 0
        if 'Storage' in techs.keys():
            self.ess_rated_power = techs['Storage'].dis_max_rated
        else:
            self.ess_rated_power = 0

        # determines how many time_series timestamps relates to the reliability target hours to cover
        self.coverage_timesteps = int(np.round(self.outage_duration_coverage / self.dt))  # integral type for indexing

        self.critical_load = params['critical load'].copy()

        self.reliability_requirement = params['critical load'].copy()
        # TODO: atm this load is only the site load, should consider aux load if included by user  --HN

        reverse = self.reliability_requirement.iloc[::-1]  # reverse the time series to use rolling function
        reverse = reverse.rolling(self.coverage_timesteps, min_periods=1).sum()*self.dt  # rolling function looks back, so reversing looks forward
        self.reliability_requirement = reverse.iloc[::-1]  # set it back the right way

        if not self.post_facto_only:
            print(f'max the system is required to store: {self.reliability_requirement.max()} kWh') if DEBUG else None
            print(f'max the system has to be able to charge bc energy req: {np.min(np.diff(self.reliability_requirement))} kW') if DEBUG else None
            print(f'max the system has to be able to discharge bc energy req: {np.max(np.diff(self.reliability_requirement))} kW') if DEBUG else None

            # add the power and energy constraints to ensure enough energy and power in the ESS for the next x hours
            # there will be 2 constraints: one for power, one for energy
            ene_min_add = Const.Constraint('ene_min', self.name, self.reliability_requirement)
            self.constraints = {'ene_min': ene_min_add}  # this should be the constraint that makes sure the next x hours have enough energy

    @staticmethod
    def rolling_sum(data, window):
        """ calculate a rolling sum of the date

        Args:
            data (DataFrame, Series): data of integers that can be added
            window (int): number of indexes to add

        Returns:

        """
        # reverse the time series to use rolling function
        reverse = data.iloc[::-1]
        # rolling function looks back, so reversing looks forward
        reverse = reverse.rolling(window, min_periods=1).sum()
        # set it back the right way
        data = reverse.iloc[::-1]
        return data

    def objective_constraints(self, variables, subs, generation, reservations=None):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            subs (DataFrame): Subset of time_series data that is being optimized
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
            reservations (Dict): power reservations from dispatch services

        Returns:
            An empty list
        """
        if not self.post_facto_only:
            try:
                pv_generation = variables['pv_out']  # time series curtailed pv optimization variable
            except KeyError:
                pv_generation = np.zeros(subs.shape[0])

            try:
                # ICE generator max rated power
                if self.n_2:
                    ice_rated_power = cvx.max(variables['n'] - 1, 0) * self.ice_rated_power
                else:
                    ice_rated_power = variables['n'] * self.ice_rated_power
            except (KeyError, AttributeError):
                ice_rated_power = 0

            # We want the minimum power capability of our DER mix in the discharge direction to be the maximum net load (load - solar)
            # to ensure that our DER mix can cover peak net load during any outage in the year
            print(f'combined max power output > {subs.loc[:, "load"].max()} kW') if DEBUG else None
            return [cvx.NonPos(cvx.max(self.critical_load.loc[subs.index].values - pv_generation) - self.ess_rated_power - ice_rated_power)]
        else:
            return super().objective_constraints(variables, subs, generation, reservations)

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.reliability_requirement.index)
        if not self.post_facto_only:
            # try:
            #     storage_energy_rating = self.storage.ene_max_rated.value
            # except AttributeError:
            #     storage_energy_rating = self.storage.ene_max_rated
            # report.loc[:, 'SOC Constraints (%)'] = self.reliability_requirement / storage_energy_rating
            report.loc[:, 'Total Outage Requirement (kWh)'] = self.reliability_requirement
        report.loc[:, 'Critical Load (kW)'] = self.critical_load
        return report

    def contribution_summary(self, technologies_keys, results):
        """ Determines that contribution from each DER type in the event of an outage

        Args:
            technologies_keys (list): list of active technologies
            results (DataFrame): dataframe that holds all the results of the optimzation

        Returns: dataframe of der's outage contribution

        """
        if not self.post_facto_only:
            outage_energy = self.reliability_requirement
            sum_outage_requirement = outage_energy.sum()  # sum of energy required to provide x hours of energy if outage occurred at every timestep

            percent_usage = {}
            contribution_arrays = {}
            if 'PV' in technologies_keys:
                # TODO: assumes we have only 1 PV
                # rolling sum of energy within a coverage_timestep window
                pv_outage_e = self.rolling_sum(results.loc[:, 'PV Maximum (kW)'], self.coverage_timesteps) * self.dt
                # try to cover as much of the outage that can be with PV energy
                net_outage_energy = outage_energy - pv_outage_e
                # pv generation might have more energy than in the outage, so dont let energy go negative
                outage_energy = net_outage_energy.clip(lower=0)

                # remove any extra energy from PV contribution
                # over_gen = -net_outage_energy.clip(upper=0)
                # pv_outage_e = pv_outage_e - over_gen
                pv_outage_e += net_outage_energy.clip(upper=0)

                # record contribution
                percent_usage.update({'PV': np.sum(pv_outage_e) / sum_outage_requirement})
                contribution_arrays.update({'PV Outage Contribution (kWh)': pv_outage_e.values})

            if 'Storage' in technologies_keys:
                ess_outage = results.loc[:, 'Aggregated State of Energy (kWh)']
                # try to cover as much of the outage that can be with the ES
                net_outage_energy = outage_energy - ess_outage
                # ESS might have more energy than in the outage, so dont let energy go negative
                outage_energy = net_outage_energy.clip(lower=0)

                # remove any extra energy from ESS contribution
                ess_outage = ess_outage + net_outage_energy.clip(upper=0)

                # record contribution
                percent_usage.update({'Storage': np.sum(ess_outage) / sum_outage_requirement})
                contribution_arrays.update({'Storage Outage Contribution (kWh)': ess_outage.values})

            if 'Diesel' in technologies_keys:
                # supplies what every energy that cannot be by pv and diesel
                # diesel_contribution is what ever is left
                percent_usage.update({'Diesel': 1 - sum(percent_usage.keys())})
                contribution_arrays.update({'Storage Outage Contribution (kWh)': outage_energy.values})

            self.contribution_df = pd.DataFrame(percent_usage, index=pd.Index(['Reliability contribution'])).T
            contribution_per_outage_df = pd.DataFrame(contribution_arrays, index=self.critical_load.index)

            return contribution_per_outage_df, self.contribution_df
        else:
            return pd.DataFrame(), pd.DataFrame()

    def load_coverage_probability(self, results_df, size_df, technology_summary_df):
        """ Creates and returns a data frame with that reports the load coverage probability of outages that last from 0 to
        OUTAGE_LENGTH hours with the DER mix described in TECHNOLOGIES

        Args:
            results_df (DataFrame): the dataframe that consoidates all results
            size_df (DataFrame): the dataframe that describes the physical capabilities of the DERs
            technology_summary_df(DataFrame): maps DER type to user inputted name that indexes the size df

        Returns: DataFrame with 2 columns - 'Outage Length (hrs)' and 'Load Coverage Probability (%)'

        Notes:  This function assumes only 1 storage (TODO) and 1 of each DER
        """
        start = time.time()

        # initialize a list to track the frequency of the results of the simulate_outage method
        frequency_simulate_outage = np.zeros(int(self.max_outage_duration / self.dt) + 1)

        # 1) simulate an outage that starts at every timestep
        # check to see if there is enough fuel generation to meet the load as offset by the amount of PV
        # generation you are confident will be delivered (usually 20% of PV forecast)
        reliability_check = self.critical_load.copy()
        demand_left = self.critical_load.copy()

        # collect information required to call simulate_outage
        tech_specs = {}
        soc = None
        # create a list of tuples with the active technologies and their names (in that order)
        technologies = []
        for name, row in technology_summary_df.iterrows():
            technologies.append((row.Type, name))

        storage_tups = [item for item in technologies if item[0] == 'Energy Storage System']
        if len(storage_tups) == 1:
            ess_properties = {'charge max': size_df.loc[storage_tups[0][1], 'Charge Rating (kW)'],
                              'discharge max': size_df.loc[storage_tups[0][1], 'Discharge Rating (kW)'],
                              'rte': size_df.loc[storage_tups[0][1], 'Round Trip Efficiency (%)'],
                              'energy cap': size_df.loc[storage_tups[0][1], 'Energy Rating (kWh)'],
                              'operation soc min': size_df.loc[storage_tups[0][1], 'Lower Limit on SOC (%)'],
                              'operation soc max': size_df.loc[storage_tups[0][1], 'Upper Limit on SOC (%)']}
            tech_specs['ess_properties'] = ess_properties
            # save the state of charge
            soc = results_df.loc[:, 'Battery SOC (%)']
        elif len(storage_tups):
            u_logger.error(f'{len(storage_tups)} storage instances included, coverage probability algorithm assumes only 1')
            return

        pv_tups = [item for item in technologies if item[0] == 'PV']
        combined_pv_max = 0  # for multiple pv
        if len(pv_tups) == 1:
            combined_pv_max = results_df.loc[:, 'PV Maximum (kW)']
            reliability_check -= self.nu * combined_pv_max
            demand_left -= combined_pv_max
        elif len(pv_tups):
            u_logger.error(f'{len(pv_tups)} pv instances included, coverage probability algorithm assumes only 1')
            return

        ice_tups = [item for item in technologies if item[0] == 'ICE']
        combined_ice_rating = 0  # for multiple ICE
        if len(ice_tups) == 1:
            if self.n_2:
                combined_ice_rating = np.max([size_df.loc[ice_tups[0][1], 'Quantity']-1, 0]) * size_df.loc[ice_tups[0][1], 'Power Capacity (kW)']
            else:
                combined_ice_rating = size_df.loc[ice_tups[0][1], 'Quantity'] * size_df.loc[ice_tups[0][1], 'Power Capacity (kW)']

            reliability_check -= combined_ice_rating
            demand_left -= combined_ice_rating
        elif len(ice_tups):
            u_logger.error(f'{len(ice_tups)} ice instances included, coverage probability algorithm assumes only 1')
            return
        end = time.time()
        u_logger.info(f'Critical Load Coverage Curve overhead time: {end - start}')
        start = time.time()
        outage_init = 0
        while outage_init < len(self.critical_load):
            if soc is not None:
                tech_specs['init_soc'] = soc.iloc[outage_init]
            longest_outage = self.simulate_outage(reliability_check.iloc[outage_init:], demand_left.iloc[outage_init:], self.max_outage_duration, **tech_specs)
            # record value of foo in frequency count
            frequency_simulate_outage[int(longest_outage / self.dt)] += 1
            # start outage on next timestep
            outage_init += 1
        # 2) calculate probabilities
        load_coverage_prob = []
        length = self.dt
        while length <= self.max_outage_duration:
            scenarios_covered = frequency_simulate_outage[int(length / self.dt):].sum()
            total_possible_scenarios = len(self.critical_load) - (length / self.dt) + 1
            percentage = scenarios_covered / total_possible_scenarios
            load_coverage_prob.append(percentage)
            length += self.dt
        # 3) build DataFrame to return
        outage_lengths = list(np.arange(0, self.max_outage_duration + self.dt, self.dt))
        outage_coverage = {'Outage Length (hrs)': outage_lengths,
                           # '# of simulations where the outage lasts up to and including': frequency_simulate_outage,
                           'Load Coverage Probability (%)': [1] + load_coverage_prob}  # first index is prob of covering outage of 0 hours (P=100%)
        end = time.time()
        u_logger.info(f'Critical Load Coverage Curve calculation time: {end - start}')
        return pd.DataFrame(outage_coverage)

    def simulate_outage(self, reliability_check, demand_left, outage_left, ess_properties=None, init_soc=None):
        """ Simulate an outage that starts with lasting only1 hour and will either last as long as MAX_OUTAGE_LENGTH
        or the iteration loop hits the end of any of the array arguments.
        Updates and tracks the SOC throughout the outage

        Args:
            reliability_check (DataFrame): the amount of load minus fuel generation and a percentage of PV generation
            demand_left (DataFrame): the amount of load minus fuel generation and all of PV generation
            init_soc (float, None): the soc of the ESS (if included in analysis) at the beginning of time t
            outage_left (int): the length of outage yet to be simulated
            ess_properties (dict): dictionary that describes the physical properties of the ess in the analysis
                includes 'charge max', 'discharge max, 'operation soc min', 'operation soc max', 'rte', 'energy cap'

        Returns: the length of the outage that starts at the beginning of the array that can be reliably covered

        """
        # base case: when to terminate recursion
        if outage_left == 0 or not len(reliability_check):
            return 0
        current_reliability_check = reliability_check.iloc[0]
        current_demand_left = demand_left.iloc[0]
        if 0 >= current_reliability_check:
            # check to see if there is space to storage energy in the ESS to save extra generation
            if ess_properties is not None and ess_properties['operation soc max'] >= init_soc:
                # the amount we can charge based on its current SOC
                soc_charge = (ess_properties['operation soc max'] - init_soc) * ess_properties['energy cap'] / (ess_properties['rte'] * self.dt)
                charge = min(soc_charge, -current_demand_left, ess_properties['charge max'])
                # update the state of charge of the ESS
                next_soc = init_soc + (charge * ess_properties['rte'] * self.dt / ess_properties['energy cap'])
            else:
                # there is no space to save the extra generation, so the ess will not do anything
                next_soc = init_soc
            # can reliably meet the outage in that timestep: CHECK NEXT TIMESTEP
        else:
            # check that there is enough SOC in the ESS to satisfy worst case
            if ess_properties is not None and 0 >= (current_reliability_check * self.gamma / ess_properties['energy cap']) - init_soc:
                # so discharge to meet the load offset by all generation
                soc_discharge = (init_soc - ess_properties['operation soc min']) * ess_properties['energy cap'] / self.dt
                discharge = min(soc_discharge, current_demand_left, ess_properties['discharge max'])
                if discharge < current_demand_left:
                    # can't discharge enough to meet demand
                    return 0
                # update the state of charge of the ESS
                next_soc = init_soc - (discharge * self.dt / ess_properties['energy cap'])
                # we can reliably meet the outage in that timestep: CHECK NEXT TIMESTEP
            else:
                # an outage cannot be reliably covered at this timestep, nor will it be covered beyond
                return 0
        # CHECK NEXT TIMESTEP
        # drop the first index of each array (so we can check the next timestep)
        next_reliability_check = reliability_check.iloc[1:]
        next_demand_left = demand_left.iloc[1:]
        return self.dt + self.simulate_outage(next_reliability_check, next_demand_left, outage_left - 1, ess_properties, next_soc)
