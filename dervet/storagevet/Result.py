"""
Result.py

"""

__author__ = 'Halley Nathwani, Thien Nyguen, Kunle Awojinrin'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'


import pandas as pd
import logging
import copy
import numpy as np
from pathlib import Path
import os
import Finances as Fin

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Result:
    """ This class serves as the later half of DER-VET's 'case builder'. It collects all optimization results, preforms
    any post optimization calculations, and saves those results to disk. If there are multiple

    """
    # these variables get read in upon importing this module
    instances = {}
    sensitivity_df = pd.DataFrame()
    sensitivity = False
    dir_abs_path = ''

    @classmethod
    def initialize(cls, results_params, df_analysis):
        """ Initialized the class with inputs that persist across all instances.

        If there are multiple runs, then set up comparison of each run.

        Args:
            results_params (Dict): user-defined inputs from the model parameter inputs
            df_analysis (DataFrame): this is a dataframe of possible sensitivity analysis instances

        Returns:

        """
        cls.instances = {}
        cls.dir_abs_path = results_params['dir_absolute_path']
        cls.csv_label = results_params['label']
        if cls.csv_label == 'nan':
            cls.csv_label = ''
        cls.sensitivity_df = df_analysis

        # data frame of all the sensitivity instances
        cls.sensitivity = (not cls.sensitivity_df.empty)
        if cls.sensitivity:
            # edit the column names of the sensitivity df to be human readable
            human_readable_names = []
            for i, col_name in enumerate(cls.sensitivity_df.columns):
                human_readable_names.append('[SP]' + col_name[0] + ' ' + col_name[1])
            # self.sens_df = pd.DataFrame()
            cls.sensitivity_df.columns = human_readable_names
            cls.sensitivity_df.index.name = 'Case Number'

    @classmethod
    def add_instance(cls, key, scenario):
        """

        Args:
            key (int): the key that corresponds to the value this instance corresponds to within the df_analysis
                dataFrame from the Params class.
            scenario (Scenario.Scenario): scenario object after optimization has run to completion

        """
        cls.template = cls(scenario)
        cls.instances.update({key: cls.template})
        cls.template.post_analysis()
        cls.template.calculate_cba()
        cls.template.save_as_csv(key, cls.sensitivity)

    def __init__(self, scenario):
        """ Initialize a Result object, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
        """
        self.opt_results = scenario.power_kw
        self.active_objects = scenario.active_objects
        self.customer_sided = scenario.customer_sided
        self.frequency = scenario.frequency
        self.dt = scenario.dt
        self.verbose_opt = scenario.verbose_opt
        self.n = scenario.n
        self.n_control = scenario.n_control
        self.mpc = scenario.mpc

        self.start_year = scenario.start_year
        self.end_year = scenario.end_year
        self.opt_years = scenario.opt_years
        self.incl_site_load = scenario.incl_site_load
        self.incl_binary = scenario.incl_binary
        self.incl_slack = scenario.incl_slack
        self.power_growth_rates = scenario.growth_rates
        self.technologies = scenario.technologies
        self.services = scenario.services
        self.predispatch_services = scenario.predispatch_services
        self.financials = scenario.financials
        self.verbose = scenario.verbose
        self.objective_values = scenario.objective_values

        # outputted DataFrames
        self.dispatch_map = pd.DataFrame()
        self.peak_day_load = pd.DataFrame()
        self.results = pd.DataFrame(index=self.opt_results.index)
        self.energyp_map = pd.DataFrame()
        self.analysis_profit = pd.DataFrame()
        self.adv_monthly_bill = pd.DataFrame()
        self.sim_monthly_bill = pd.DataFrame()
        self.monthly_data = pd.DataFrame()
        self.deferral_dataframe = scenario.deferral_df
        self.technology_summary = None
        self.demand_charges = None

    def post_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        TODO: [multi-tech] a lot of this logic will have to change with multiple technologies
        """

        print("Performing Post Optimization Analysis...") if self.verbose else None

        # add other helpful information to a RESULTS DATAFRAME
        self.results.loc[:, 'Total Load (kW)'] = self.opt_results['load']  # this is the site load, if included
        self.results.loc[:, 'Total Generation (kW)'] = 0
        # collect all storage power to handle multiple storage technologies, similar to total generation
        self.results.loc[:, 'Total Storage Power (kW)'] = 0
        self.results.loc[:, 'Aggregated State of Energy (kWh)'] = 0

        # collect results from technologies
        tech_type = []
        tech_name = []

        #  output to timeseries_results.csv
        for name, tech in self.technologies.items():
            if 'Deferral' not in self.predispatch_services.keys() or len(self.services.keys()):
                # if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
                # then we did not optimize (skipped the optimization loop) NOTE - INVERSE OF CONDITIONAL ON LINE 547 in STORAGEVET\SCENARIO.PY
                report_df = tech.timeseries_report()
                self.results = pd.concat([report_df, self.results], axis=1)

                if name == 'PV':
                    self.results.loc[:, 'Total Generation (kW)'] += self.results['PV Generation (kW)']
                if name == 'ICE':
                    self.results.loc[:, 'Total Generation (kW)'] += self.results['ICE Generation (kW)']
                if name == 'Storage':
                    self.results.loc[:, 'Total Storage Power (kW)'] += self.results[tech.name + ' Power (kW)']
                    self.results.loc[:, 'Aggregated State of Energy (kWh)'] += self.results[tech.name + ' State of Energy (kWh)']
            tech_name.append(tech.name)
            tech_type.append(tech.type)

        self.technology_summary = pd.DataFrame({'Type': tech_type}, index=pd.Index(tech_name, name='Name'))

        # collect results from each value stream
        for service in self.services.values():
            report_df = service.timeseries_report()
            self.results = pd.concat([self.results, report_df], axis=1)
            report = service.monthly_report()
            self.monthly_data = pd.concat([self.monthly_data, report], axis=1, sort=False)

        for pre_dispatch in self.predispatch_services.values():
            report_df = pre_dispatch.timeseries_report()
            self.results = pd.concat([self.results, report_df], axis=1)
            report = pre_dispatch.monthly_report()
            self.monthly_data = pd.concat([self.monthly_data, report], axis=1, sort=False)

        # assumes the orginal net load only does not contain the Storage system
        self.results.loc[:, 'Original Net Load (kW)'] = self.results['Total Load (kW)']
        self.results.loc[:, 'Net Load (kW)'] = self.results['Total Load (kW)'] - self.results['Total Generation (kW)'] - self.results['Total Storage Power (kW)']

        if 'DCM' in self.active_objects['value streams']:
            self.demand_charges = self.services['DCM'].tariff

        if 'retailTimeShift' in self.active_objects['value streams']:
            energy_price = self.results.loc[:, 'Energy Price ($/kWh)'].to_frame()
            energy_price.loc[:, 'date'] = self.opt_results.index.date
            energy_price.loc[:, 'hour'] = (self.opt_results.index + pd.Timedelta('1s')).hour + 1
            energy_price = energy_price.reset_index(drop=True)
            self.energyp_map = energy_price.pivot_table(values='Energy Price ($/kWh)', index='hour', columns='date')

        if "DA" in self.services.keys():
            energy_price = self.results.loc[:, 'DA Price Signal ($/kWh)'].to_frame()
            energy_price.loc[:, 'date'] = self.opt_results.index.date
            energy_price.loc[:, 'hour'] = (self.opt_results.index + pd.Timedelta('1s')).hour + 1
            energy_price = energy_price.reset_index(drop=True)
            self.energyp_map = energy_price.pivot_table(values='DA Price Signal ($/kWh)', index='hour', columns='date')

        if 'Deferral' in self.active_objects['value streams']:
            # these try to capture the import power to the site pre and post storage
            if self.deferral_dataframe is None:
                self.results.loc[:, 'Pre-storage Net Power (kW)'] = self.results['Total Load (kW)'] - self.results['Total Generation (kW)']
                self.results.loc[:, 'Pre-storage Net Power (kW)'] += self.results['Deferral Load (kW)']
                self.results.loc[:, 'Post-storage Net Power (kW)'] = self.results['Pre-storage Net Power (kW)']
                for name, tech in self.technologies.items():
                    self.results.loc[:, 'Post-storage Net Power (kW)'] = self.results['Post-storage Net Power (kW)'] - self.results[tech.name + ' Power (kW)']

        # create DISPATCH MAP dictionary to handle multiple storage technologies
        if self.deferral_dataframe is None:
            self.dispatch_map = {}  # TODO: this needs to be initialized witin the __init__ function  --HN
            for name, tech in self.technologies.items():
                if name == 'Storage':
                    if self.verbose:
                        constraints_df = self.technologies['Storage'].verbose_results()
                        self.opt_results = pd.concat([constraints_df, self.opt_results], axis=1)
                    dispatch = self.results.loc[:, tech.name + ' Power (kW)'].to_frame()
                    dispatch.loc[:, 'date'] = self.opt_results.index.date
                    dispatch.loc[:, 'hour'] = (self.opt_results.index + pd.Timedelta('1s')).hour + 1
                    dispatch = dispatch.reset_index(drop=True)

                    self.dispatch_map = dispatch.pivot_table(values=tech.name + ' Power (kW)', index='hour', columns='date')

        u_logger.debug("Finished post optimization analysis")

    def calculate_cba(self):
        """ Calls all finacial methods that will result in a series of dataframes to describe the cost benefit analysis for the
        case in question.

        """
        value_streams = {**self.services, **self.predispatch_services}
        if 'Deferral' not in self.predispatch_services.keys() or len(self.services.keys()):
            # if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
            # then we did not optimize (skipped the optimization loop) NOTE - INVERSE OF CONDITIONAL ON LINE 547 in STORAGEVET\SCENARIO.PY
            self.financials.preform_cost_benefit_analysis(self.technologies, value_streams, self.results)

    def validation(self):
        """ Goes through results and returns a summary CSV.

        Returns (DataFrame): Is a quick summary, labeled in short hand, each row is a different value

        """

        df = pd.DataFrame()
        for name, tech in self.technologies.items():
            if name == 'Battery' or name == 'CAES':
                # the names need to be updated accordingly to how each storage technology report in their own timeseries_report method
                temp = pd.DataFrame({tech.name + ' SOE Min': self.results[tech.name + ' ' + name + ' State of Energy (kWh)'].min(),
                                    tech.name + ' SOE Max': self.results[tech.name + ' ' + name + ' State of Energy (kWh)'].max(),
                                    tech.name + ' Charge Min': self.results[tech.name + ' ' + name + ' Charge (kW)'].min(),
                                    tech.name + ' Charge Max': self.results[tech.name + ' ' + name + ' Charge (kW)'].max(),
                                    tech.name + ' Discharge Min': self.results[tech.name + ' ' + name + ' Discharge (kW)'].min(),
                                    tech.name + ' Discharge Max': self.results[tech.name + ' ' + name + ' Discharge (kW)'].max()},
                                    index=pd.Index(['Value']))
                df = pd.concat([df, temp], axis=1, ignore_index=False)

        return df.T

    def save_as_csv(self, instance_key, sensitivity=False):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        Args:
            instance_key (int): string of the instance value that corresponds to the Params instance that was used for
                this simulation.
            sensitivity (boolean): logic if sensitivity analysis is active. If yes, save_path should create additional
                subdirectory

        Prints where the results have been saved when completed.
        """
        if sensitivity:
            savepath = self.dir_abs_path + "\\" + str(instance_key)
        else:
            savepath = self.dir_abs_path
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.results.sort_index(axis=1, inplace=True)  # sorts by column name alphabetically
        self.results.to_csv(path_or_buf=Path(savepath, 'timeseries_results' + self.csv_label + '.csv'))
        if self.customer_sided:
            self.financials.billing_period_bill.to_csv(path_or_buf=Path(savepath, 'adv_monthly_bill' + self.csv_label + '.csv'))
            self.financials.monthly_bill.to_csv(path_or_buf=Path(savepath, 'simple_monthly_bill' + self.csv_label + '.csv'))
            if 'DCM' in self.active_objects['value streams']:
                self.demand_charges.to_csv(path_or_buf=Path(savepath, 'demand_charges' + self.csv_label + '.csv'))

        self.peak_day_load.to_csv(path_or_buf=Path(savepath, 'peak_day_load' + self.csv_label + '.csv'))
        self.dispatch_map.to_csv(path_or_buf=Path(savepath, 'dispatch_map' + self.csv_label + '.csv'))

        if 'Deferral' in self.predispatch_services.keys():
            self.deferral_dataframe.to_csv(path_or_buf=Path(savepath, 'deferral_results' + self.csv_label + '.csv'))

        if 'DA' in self.services.keys() or 'retailTimeShift' in self.services.keys():
            self.energyp_map.to_csv(path_or_buf=Path(savepath, 'energyp_map' + self.csv_label + '.csv'))
        self.technology_summary.to_csv(path_or_buf=Path(savepath, 'technology_summary' + self.csv_label + '.csv'))

        # add other services that have monthly data here, if we want to save its monthly financial data report
        self.monthly_data.to_csv(path_or_buf=Path(savepath, 'monthly_data' + self.csv_label + '.csv'))

        ###############################
        # PRINT FINALCIAL/CBA RESULTS #
        ###############################
        self.financials.pro_forma.to_csv(path_or_buf=Path(savepath, 'pro_forma' + self.csv_label + '.csv'))
        self.financials.npv.to_csv(path_or_buf=Path(savepath, 'npv' + self.csv_label + '.csv'))
        self.financials.cost_benefit.to_csv(path_or_buf=Path(savepath, 'cost_benefit' + self.csv_label + '.csv'))
        self.financials.payback.to_csv(path_or_buf=Path(savepath, 'payback' + self.csv_label + '.csv'))

        if self.verbose:
            self.validation().to_csv(path_or_buf=Path(savepath, 'validation' + self.csv_label + '.csv'))
            self.opt_results.to_csv(path_or_buf=Path(savepath, 'opt_results' + self.csv_label + '.csv'))
            self.objective_values.to_csv(path_or_buf=Path(savepath, 'objective_values' + self.csv_label + '.csv'))
        print('Results have been saved to: ' + savepath)

    @classmethod
    def sensitivity_summary(cls):
        """ Loop through all the Result instances to build the dataframe capturing the important financial results
        and unique sensitivity input parameters for all instances.
            Then save the dataframe to a csv file.

        """
        if cls.sensitivity:
            for key, results_object in cls.instances.items():
                if not key:
                    for npv_col in results_object.financials.npv.columns:
                        cls.sensitivity_df.loc[:, npv_col] = 0
                this_npv = results_object.financials.npv.reset_index(drop=True, inplace=False)
                this_npv.index = pd.RangeIndex(start=key, stop=key + 1, step=1)
                cls.sensitivity_df.update(this_npv)

            cls.sensitivity_df.to_csv(path_or_buf=Path(cls.dir_abs_path, 'sensitivity_summary.csv'))

    @classmethod
    def proforma_df(cls, instance=0):
        """ Return the financial pro_forma for a specific instance

        """
        return cls.instances[instance].financials.pro_forma

    # # TODO Taken from svet_outputs. ploting functions haven't been used - EG + YY
    #
    # def plotly_stacked(self, p1_y1_bar=None, p1_y2=None, p2_y1=None, price_col=None, sep_ene_plot=True, filename=None,  year=None, start=None, end=None):
    #
    #     deferral = self.inputs['params']['Deferral']
    #
    #     if p1_y1_bar is None:
    #         p1_y1_bar = ['ch', 'dis']
    #     if p1_y2 is None:
    #         p1_y2 = ['ene']
    #     if p2_y1 is None:
    #         p2_y1 = []
    #         if deferral:
    #             p2_y1_line = ['net_deferral_import_power', 'pretech_deferral_import_power']
    #         else:
    #             p2_y1_line = ['net_import_power', 'pretech_import_power']
    #
    #         p2_y1_load = ['load']
    #         if deferral:
    #             p2_y1_load += ['deferral_load']
    #
    #         p2_y1_gen = []
    #         if all(self.results['ac_gen'] != 0):
    #             p2_y1_gen += ['ac_gen']
    #         if all(self.results['dc_gen'] != 0):
    #             p2_y1_gen += ['dc_gen']
    #         if deferral:
    #             p2_y1_gen = ['deferral_gen']
    #         p2_y1 += p2_y1_line + p2_y1_load + p2_y1_gen
    #
    #     # get price columns
    #     if price_col is None:
    #         price_col = []
    #
    #     # TODO do this smarter
    #     p1_y1_arrow = []
    #     price_col_kwh = []
    #     price_col_kw = []
    #     if self.inputs['params']['SR']:
    #         p1_y1_arrow += ['sr_d', 'sr_c']
    #         price_col_kw += ['p_sr']
    #     if self.inputs['params']['FR']:
    #         p1_y1_arrow += ['regu_d', 'regu_c', 'regd_d', 'regd_c']
    #         price_col_kw += ['p_regu', 'p_regd']
    #     if self.inputs['params']['DA']:
    #         price_col_kwh += ['p_da']
    #     if self.inputs['params']['retailTimeShift']:
    #         price_col_kwh += ['p_energy']
    #     price_col += price_col_kwh + price_col_kw
    #     price_plot_kw = len(price_col_kw) > 0  # flag for subplot logic
    #     price_plot_kwh = len(price_col_kwh) > 0  # flag for subplot logic
    #
    #
    #     # convert $/kW to $/MW
    #     price_data = self.financials.fin_inputs[price_col] * 1000
    #
    #     # merge price data with load data
    #     col_names = ['year'] + p1_y1_bar+p1_y1_arrow + p1_y2 + p2_y1
    #     plot_results = pd.merge(self.results[col_names], price_data, left_index=True, right_index=True, how='left')
    #
    #     # represent charging as negative
    #     neg_cols = ['ch']
    #
    #     # combine FR columns to reg up and reg down
    #     if self.inputs['params']['FR']:
    #         plot_results['reg_up'] = plot_results['regu_d'] + plot_results['regu_c']
    #         plot_results['reg_down'] = plot_results['regd_d'] + plot_results['regd_c']
    #         neg_cols += ['reg_down']
    #         for col in ['regu_d', 'regu_c', 'regd_d', 'regd_c']:
    #             p1_y1_arrow.remove(col)
    #         p1_y1_arrow += ['reg_up', 'reg_down']
    #     plot_results[neg_cols] = -plot_results[neg_cols]
    #
    #     # subset plot_results based on parameters
    #     if year is not None:
    #         plot_results = plot_results[plot_results.year == year]
    #     if start is not None:
    #         plot_results = plot_results[plot_results.index > start]
    #     if end is not None:
    #         plot_results = plot_results[plot_results.index <= end]
    #
    #     # round small numbers to zero for easier viewing
    #     plot_results = plot_results.round(decimals=6)
    #
    #     # add missing rows to avoid interpolation
    #     plot_results = Lib.fill_gaps(plot_results)
    #
    #     # create figure
    #     fig = py.tools.make_subplots(rows=2 + sep_ene_plot + price_plot_kwh + price_plot_kw, cols=1, shared_xaxes=True, print_grid=False)
    #
    #     fig['layout'].update(barmode='relative')  # allows negative values to have negative bars
    #     # fig['layout'].update(barmode='overlay')
    #
    #     # add ch and discharge
    #     for col in p1_y1_bar:
    #         trace = py.graph_objs.Bar(x=plot_results.index, y=plot_results[col], name=col, offset=pd.Timedelta(self.dt/2, unit='h'))
    #         fig.append_trace(trace, 1, 1)
    #     battery_power = plot_results['dis'] + plot_results['ch']
    #
    #     # add capacity commitments such as reg up and reg down as error bars
    #     colors = py.colors.DEFAULT_PLOTLY_COLORS
    #     for i, col in enumerate(p1_y1_arrow):
    #         y0_txt = [str(y0) for y0 in plot_results[col]]
    #         trace = py.graph_objs.Bar(x=plot_results.index, y=battery_power*0, base=battery_power, name=col, offset=pd.Timedelta(self.dt/2, unit='h'), text=y0_txt,
    #                                   hoverinfo='x+text+name', marker=dict(color='rgba(0, 0, 0, 0)'), hoverlabel=dict(bgcolor=colors[i+len(p1_y1_bar)]),
    #                                   showlegend=False,  # TODO hiding until figure out how to include error bars in legend
    #                                   error_y=dict(visible=True, symmetric=False, array=plot_results[col], type='data', color=colors[i+len(p1_y1_bar)]))
    #         fig.append_trace(trace, 1, 1)
    #
    #     # other methods I tried instead of using error bars for capacity commitments
    #
    #     # for i, col in enumerate(p1_y1_arrow):
    #     #     y0_txt = [str(y0) for y0 in plot_results[col]]
    #     #     trace = ff.create_quiver(x=plot_results.index, y=battery_power.values, u=plot_results.index, v=plot_results[col].values)
    #     #     fig.append_trace(trace, 1, 1)
    #
    #     # colors = py.colors.DEFAULT_PLOTLY_COLORS
    #     # for i, col in enumerate(p1_y1_arrow):
    #     #     y0_txt = [str(y0) for y0 in plot_results[col]]
    #     #     trace = py.graph_objs.Bar(x=plot_results.index, y=plot_results[col], base=battery_power, name=col, offset=-55*60*case.dt*1e3/2, width=10000,
    #     #                               marker=dict(color='rgba(0, 0, 0, 0)', line=dict(width=2, color=colors[i+2])), text=y0_txt, hoverinfo='x+text+name')
    #     #     fig.append_trace(trace, 1, 1)
    #
    #     # for col in p1_y1_arrow:
    #     #     battery_power = plot_results['dis'] - plot_results['ch']
    #     #     trace = py.graph_objs.Scatter(x=plot_results.index-pd.Timedelta(-case.dt/2, unit='h'), y=battery_power+plot_results[col], name=col, mode='markers', line=dict(shape='vh'))
    #     #     fig.append_trace(trace, 1, 1)
    #
    #     if sep_ene_plot:
    #         # add separate energy plot
    #         for col in p1_y2:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='linear'), name=col, mode='lines+markers')
    #             fig.append_trace(trace, 2, 1)
    #     else:
    #         # add energy on second y axis to charge and discharge
    #         for col in p1_y2:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='linear'), name=col, yaxis='y5', mode='lines+markers')
    #             fig.append_trace(trace, 1, 1)
    #             names = []
    #             for i in fig.data:
    #                 names += [i['name']]
    #             fig.data[names.index('ene')].update(yaxis='y5')
    #
    #     # system power plot
    #     p2_y1_load_val = np.zeros(len(plot_results))
    #     for col in p2_y1_load:
    #         p2_y1_load_val += plot_results[col]
    #         y0_txt = [str(y0) for y0 in plot_results[col]]
    #         trace = py.graph_objs.Scatter(x=plot_results.index, y=copy.deepcopy(p2_y1_load_val), line=dict(shape='vh'), name=col, fill='tonexty',mode='none',
    #                                       text=y0_txt, hoverinfo='x+text+name')
    #         fig.append_trace(trace, 2 + sep_ene_plot, 1)
    #     p2_y1_gen_val = np.zeros(len(plot_results))
    #     for col in p2_y1_gen:
    #         p2_y1_gen_val -= plot_results[col]
    #         y0_txt = [str(y0) for y0 in plot_results[col]]
    #         trace = py.graph_objs.Scatter(x=plot_results.index, y=p2_y1_gen_val, line=dict(shape='vh'), name=col, fill='tonexty', mode='none',
    #                                       text=y0_txt, hoverinfo='x+text+name')
    #         fig.append_trace(trace, 2 + sep_ene_plot, 1)
    #     for col in p2_y1_line:
    #         trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='vh'), name=col, mode='lines+markers')
    #         fig.append_trace(trace, 2 + sep_ene_plot, 1)
    #
    #     # price plot
    #     if price_plot_kw:
    #         for col in price_col_kw:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='vh'), name=col)
    #             fig.append_trace(trace, 3 + sep_ene_plot, 1)
    #     if price_plot_kwh:
    #         for col in price_col_kwh:
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], line=dict(shape='vh'), name=col, yaxis='y6')
    #             fig.append_trace(trace, 3 + price_plot_kw + sep_ene_plot, 1)
    #
    #     # axis labels
    #     fig['layout']['xaxis1'].update(title='Time')
    #     fig['layout']['yaxis1'].update(title='Power (kW)')
    #     if not sep_ene_plot:
    #         fig['layout']['yaxis2'].update(title='Power (kW)')
    #         if price_plot_kw:
    #             fig['layout']['yaxis3'].update(title='Price ($/MW)')
    #         if price_plot_kwh:
    #             fig['layout']['yaxis4'].update(title='Price ($/MWh)')
    #         fig['layout']['yaxis5'] = dict(overlaying='y1', anchor='x1', side='right', title='Energy (kWh)')
    #     else:
    #         fig['layout']['yaxis2'].update(title='Energy (kWh)')
    #         fig['layout']['yaxis3'].update(title='Power (kW)')
    #         if price_plot_kw:
    #             fig['layout']['yaxis4'].update(title='Price ($/MW)')
    #         if price_plot_kwh:
    #             fig['layout']['yaxis5'].update(title='Price ($/MWh)')
    #
    #     # move legend to middle (plotly does not support separate legends for subplots as of now)
    #     fig['layout']['legend'] = dict(y=0.5, traceorder='normal')
    #
    #     if filename is None:
    #         filename = self.name + '_plot.html'
    #
    #     py.offline.plot(fig, filename=filename)
    #
    # def plotly_groupby(self, group='he'):
    #     """ Plot Results averaged to a certain group column
    #
    #     Args:
    #         case (Case): case object
    #         group (string, list): columns in case.results to group on
    #
    #     """
    #     yrs = self.opt_years
    #
    #     fig = py.tools.make_subplots(rows=len(yrs), cols=1, shared_xaxes=True, print_grid=False, subplot_titles=[yr.year for yr in yrs])
    #
    #     colors = py.colors.DEFAULT_PLOTLY_COLORS
    #
    #     for i, yr in enumerate(yrs):
    #         plot_results = self.power_kw[self.power_kw.year == yr].groupby(group).mean()
    #         neg_cols = ['ch']
    #         if self.inputs['params']['FR']:
    #             plot_results['reg_up'] = plot_results['regu_d'] + plot_results['regu_c']
    #             plot_results['reg_down'] = plot_results['regd_d'] + plot_results['regd_c']
    #             neg_cols += ['reg_down']
    #
    #         plot_results[neg_cols] = -plot_results[neg_cols]
    #
    #         plot_cols = ['ch', 'dis', 'reg_up', 'reg_down', 'load']
    #         for ii, col in enumerate(plot_cols):
    #             trace = py.graph_objs.Scatter(x=plot_results.index, y=plot_results[col], name=col, mode='lines+markers',
    #                                           line=dict(color=colors[ii]))
    #             fig.append_trace(trace, i+1, 1)
    #     py.offline.plot(fig)
    #
    # # older plotting function
    # def plotly_case(self, y1_col=None, y2_col=None, binary=False, filename=None, start=None, end=None):
    #     """ Generic plotly function for case object (depreciated by plotly_stacked)
    #
    #     Args:
    #         case (Case): case object
    #         y1_col (list, optional): column names in case.result to plot on y1 axis
    #         y2_col (list, optional): column names in case.result to plot on y2 axis
    #         binary (bool, optional): Flag to add binary variables
    #         filename (str, optional): where to save file (must end in .html)
    #         start (date-like, optional): start timestamp to subset data (exclusive)
    #         end (date-like, optional): end timestamp to subset data (inclusive)
    #
    #     """
    #     if y1_col is None:
    #         y1_col = ['ch', 'dis', 'net_import_power', 'pretech_import_power']
    #     if y2_col is None:
    #         y2_col = ['ene']
    #
    #     limit_cols = []
    #
    #     plot_results = copy.deepcopy(self.results)
    #     predispatch = list(self.predispatch_services)
    #
    #     if 'Deferral' in predispatch:
    #         plot_results['deferral_max_import'] = self.predispatch_services['Deferral'].deferral_max_import
    #         plot_results['deferral_max_export'] = self.predispatch_services['Deferral'].deferral_max_export
    #         y1_col += ['deferral_max_import', 'deferral_max_export']
    #         limit_cols += ['deferral_max_import', 'deferral_max_export']
    #     if 'Volt' in predispatch:
    #         plot_results['inv_max'] = self.inputs['params']['inv_max']
    #         y1_col += ['vars_load', 'inv_max']
    #         limit_cols += ['inv_max']
    #     if 'Backup' in predispatch:
    #         y2_col += ['backup_energy']
    #
    #     # TODO do this smarter
    #     if self.inputs['params']['SR']:
    #         y1_col += ['sr_d', 'sr_c']
    #     if self.inputs['params']['FR']:
    #         y1_col += ['regu_d', 'regu_c', 'regd_d', 'regd_c']
    #
    #     if binary:  # or (binary is None and case.inputs['params']['binary']):
    #         y1_col += ['on_c', 'on_d']
    #
    #     if start is not None:
    #         plot_results = plot_results[plot_results.index > start]
    #     if end is not None:
    #         plot_results = plot_results[plot_results.index <= end]
    #
    #     # round small numbers to zero for easier viewing
    #     plot_results = plot_results.round(decimals=6)
    #
    #     # add missing rows to avoid interpolation
    #     plot_results = Lib.fill_gaps(plot_results)
    #
    #     fig = plot_results[y1_col+y2_col].iplot(kind='line', mode='lines', interpolation='vh', asFigure=True, yTitle='Power (kW)', secondary_y=y2_col)
    #
    #     names = []
    #     for i in fig.data:
    #         names += [i['name']]
    #
    #     if 'ene' in y2_col:
    #         fig.layout.yaxis2.title = 'Energy (kWh)'
    #
    #         fig.data[names.index('ene')].line.shape = 'linear'
    #
    #     for n in limit_cols:
    #         fig.data[names.index(n)].line.dash = 'dot'
    #         fig.data[names.index(n)].line.width = 1
    #
    #     if filename is None:
    #         filename = self.name + '_plot.html'
    #
    #     py.offline.plot(fig, filename=filename)
    #
    # def plotly_prices(self, y1_col=None, price_col=None, filename=None, start=None, end=None):
    #     """ Generic plotly price function for case object (depreciated by plotly_stacked)
    #
    #     Args:
    #         case (Case): case object
    #         y1_col (list, optional): column names in case.result to plot on y1 axis
    #         price_col (list, optional): price column names in case.financials.fin_inputs to plot on y2 axis
    #         filename (str, optional): where to save file (must end in .html)
    #         start (date-like, optional): start timestamp to subset data (exclusive)
    #         end (date-like, optional): end timestamp to subset data (inclusive)
    #
    #     """
    #     if y1_col is None:
    #         y1_col = ['ch', 'dis', 'pretech_import_power', 'net_import_power']
    #     if price_col is None:
    #         price_col = []
    #
    #     # TODO do this smarter
    #     if self.inputs['params']['SR']:
    #         y1_col += ['sr_d', 'sr_c']
    #         price_col += ['p_sr']
    #     if self.inputs['params']['FR']:
    #         y1_col += ['regu_d', 'regu_c', 'regd_d', 'regd_c']
    #         price_col += ['p_regu', 'p_regd']
    #     if self.inputs['params']['DA']:
    #         price_col += ['p_da']
    #     if self.inputs['params']['retailTimeShift']:
    #         price_col += ['p_energy']
    #
    #     price_data = self.financials.fin_inputs[price_col]*1000
    #
    #     plot_results = pd.merge(self.results[y1_col], price_data, left_index=True, right_index=True, how='left')
    #
    #     if start is not None:
    #         plot_results = plot_results[plot_results.index > start]
    #     if end is not None:
    #         plot_results = plot_results[plot_results.index <= end]
    #
    #     # round small numbers to zero for easier viewing
    #     plot_results = plot_results.round(decimals=6)
    #
    #     # add missing rows to avoid interpolation
    #     plot_results = Lib.fill_gaps(plot_results)
    #
    #     fig = plot_results[y1_col+price_col].iplot(kind='line', mode='lines', interpolation='vh', asFigure=True, yTitle='Power (kW)', secondary_y=list(plot_results[price_col]))
    #     fig.layout.yaxis2.title = 'Price ($/MWh)'
    #
    #     if filename is None:
    #         filename = self.name + '_plot.html'
    #
    #     py.offline.plot(fig, filename=filename)
    #
    # def plot_results(self, start_data=0, end_data=None, save=False):
    #     """ Plot the energy and demand charges before and after the storage acts and validates results by checking if
    #     the change in SOC is accounted for by AC power. Additionally saves plots in the results folder created when
    #     initialised.
    #
    #     """
    #     if not end_data:
    #         end_data = self.inputs['time_series'].index.size
    #     charging_opt_var = ['reg']
    #     results = self.financials.obj_val
    #     os.makedirs(self.dir_abs_path)
    #     plt.rcParams['figure.dpi'] = 300
    #     plt.rcParams['figure.figsize'] = [12, 6.75]
    #     plt.figure()
    #
    #     load = self.power_kw['site_load']
    #     bulk_power = self.power_kw['dis'] - self.power_kw['ch']
    #     pv = self.power_kw['PV_gen']
    #     net_power = load - bulk_power - pv  # at the POC
    #     soc = self.power_kw['ene'] / self.technologies['Storage'].ene_max_rated
    #     soc_diff = soc.diff()
    #     ac_power = copy.deepcopy(bulk_power)
    #     for serv in self.services.values():
    #         temp_serv_p = serv.ene_results['ene'] / self.dt
    #         ac_power = ac_power + temp_serv_p
    #
    #     plt.plot(load.index[start_data:end_data], load[start_data:end_data])
    #     plt.plot(net_power.index[start_data:end_data], net_power[start_data:end_data])
    #     plt.plot(pv.index[start_data:end_data], pv[start_data:end_data])
    #     plt.legend(['Site Load', 'Net Power', 'PV'])
    #     plt.ylabel('Power (kW)')
    #     if save:
    #         plt.savefig(self.dir_abs_path + 'net_power.png')
    #     plt.close()
    #
    #     plt.plot(bulk_power.index[start_data:end_data], bulk_power[start_data:end_data])
    #     plt.title('Storage Power')
    #     plt.ylabel('Power (kW)')
    #     if save:
    #         plt.savefig(self.dir_abs_path + 'sto_power.png')
    #     plt.close()
    #
    #     plt.plot(soc.index[start_data:end_data], soc[start_data:end_data])
    #     plt.title('Storage State of Charge')
    #     plt.ylabel('Power (kW)')
    #     if save:
    #         plt.savefig(self.dir_abs_path + 'state_of_charge.png')
    #     plt.close()
    #
    #     # plot the energy and demand charges before and after the storage acts
    #     # width = .2
    #     # plt.figure()
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) + 2 * width, case.monthly_bill.loc[:, 'energy_charge'], width)
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) + width, case.monthly_bill.loc[:, 'original_energy_charge'], width)
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) - 2 * width, case.monthly_bill.loc[:, 'demand_charge'], width)
    #     # plt.bar(list(list(zip(*case.monthly_bill.index.values))[0]) - width, case.monthly_bill.loc[:, 'original_demand_charge'], width)
    #     # plt.title('Monthly energy and demand charges')
    #     # plt.legend(['Energy Charges', 'Original Energy Charges', 'Demand Charges', 'Original Demand Charges'])
    #     # plt.xlabel('Month')
    #     # plt.ylabel('$')
    #     # plt.draw()
    #     # plt.savefig(case.dir_abs_path + 'charges.png')
    #
    #     # Validate Results by checking if the change in SOC is accounted for by AC power
    #     plt.scatter(soc_diff, ac_power)
    #     plt.xlabel('delta SOC')
    #     plt.ylabel('AC Storage Power')
    #     plt.draw()
    #     if save:
    #         plt.savefig(self.dir_abs_path + 'SOCvkW.png')
