"""
Scenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""

__author__ = 'Halley Nathwani, Evan Giarta, Thien Nygen'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'mevans@epri.com']
__version__ = '0.1.1'


import storagevet

from TechnologiesDER.BatterySizing import BatterySizing
from TechnologiesDER.CAESSizing import CAESSizing
from TechnologiesDER.CurtailPVSizing import CurtailPVSizing
from TechnologiesDER.ICESizing import ICESizing
from ValueStreamsDER.Reliability import Reliability

from storagevet.Scenario import Scenario

from cbaDER import CostBenefitAnalysis

import logging

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class ScenarioSizing(Scenario):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario.

        Args:
            input_tree (Dict): Dict of input attributes such as time_series, params, and monthly_data

        """
        Scenario.__init__(self, input_tree)

        self.predispatch_service_inputs_map.update({'Reliability': input_tree.Reliability})

        self.sizing_optimization = False

        u_logger.info("ScenarioDER initialized ...")

    def init_financials(self, finance_inputs):
        """ Initializes the financial class with a copy of all the price data from timeseries, the tariff data, and any
         system variables required for post optimization analysis.

         Args:
             finance_inputs (Dict): Financial inputs

        """

        self.financials = CostBenefitAnalysis(finance_inputs)
        u_logger.info("Finished adding Financials...")

    def check_if_sizing_ders(self):
        """ This method will iterate through the initialized DER instances and return a logical OR of all of their
        'being_sized' methods.

        Returns: True if ANY DER is getting sized

        """
        for der in self.technologies.values():
            try:
                solve_for_size = der.being_sized()
            except AttributeError:
                solve_for_size = False
            if solve_for_size:
                return True
        return False

    def add_technology(self):
        """ Reads params and adds technology. Each technology gets initialized and their physical constraints are found.

        """
        ess_action_map = {
            'Battery': BatterySizing,
            'CAES': CAESSizing
        }

        for storage in ess_action_map.keys():  # this will cause merging errors -HN
            inputs = self.technology_inputs_map[storage]
            if inputs is not None:
                tech_func = ess_action_map[storage]
                self.technologies["Storage"] = tech_func('Storage', self.power_kw['opt_agg'], inputs)
            u_logger.info("Finished adding storage...")

        generator_action_map = {
            'PV': CurtailPVSizing,
            'ICE': ICESizing
        }

        for gen in generator_action_map.keys():
            inputs = self.technology_inputs_map[gen]
            if inputs is not None:
                tech_func = generator_action_map[gen]
                new_gen = tech_func(gen, inputs)
                new_gen.estimate_year_data(self.opt_years, self.frequency)
                self.technologies[gen] = new_gen
        u_logger.info("Finished adding generators...")

        self.sizing_optimization = self.check_if_sizing_ders()

    def add_services(self):
        """ Reads through params to determine which services are turned on or off. Then creates the corresponding
        service object and adds it to the list of services. Also generates a list of growth functions that apply to each
        service's timeseries data (to be used when adding growth data).

        Notes:
            This method needs to be applied after the technology has been initialized.
            ALL SERVICES ARE CONNECTED TO THE TECH

        """

        if self.predispatch_service_inputs_map['Reliability']:
            u_logger.info("Using: Reliability")
            inputs = self.predispatch_service_inputs_map['Reliability']
            new_service = Reliability(inputs, self.technologies, self.power_kw, self.dt)
            new_service.estimate_year_data(self.opt_years, self.frequency)
            self.predispatch_services['Reliability'] = new_service
            self.predispatch_service_inputs_map.pop('Reliability')

        super().add_services()

    def optimize_problem_loop(self, annuity_scalar=1):
        """This function selects on opt_agg of data in self.time_series and calls optimization_problem on it. We determine if the
        optimization will be sizing and calculate a lifetime project NPV multiplier to pass into the optimization problem

        Args:
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        """
        if self.sizing_optimization:
            annuity_scalar = self.financials.annuity_scalar(self.start_year, self.end_year, self.opt_years)

        super().optimize_problem_loop(annuity_scalar)
