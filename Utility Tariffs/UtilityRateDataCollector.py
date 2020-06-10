import pandas as pd
import requests
import json
import pickle
import numpy as np

# git data url
git_data_url = "https://raw.githubusercontent.com/tammydiehl/ResidentialRenewablesAnalyzer/master/"  #residential_load_pickles/BASE/USA_AK_Anchorage.Intl.AP.702730_TMY3_BASE.pkl

# these seems to be the most interesting tidbits that we are interested in
poi = ['name', 'description', 'source', 'dgrules', 'energyratestructure', 'energyweekdayschedule', 'energyweekendschedule', 'energycomments', 'demandrateunit', 'demandunits', 'fixedchargefirstmeter', 'fixedchargeunits', 'flatdemandunit']

# parameters for OpenEI tariff request
paramets = {
        "version": "latest",
        "format": "json",
        "api_key": "snVIu7FREbmucvgeZjPgsumRjdCCqQtwHd0QJsmb",
        "getpage": "5d5582305457a3e01135a086",   # this will be populated by user input. 5d5582305457a3e01135a086
        "detail": "full",
        "limit": 500
    }


### Use this plant data to get a mapping of electric utilities to EIA IDs and the states that the utility is in
# eia_data = pd.read_excel("/Users/jhthompson12/Desktop/EIA923_Schedules_2_3_4_5_M_02_2020_20APR2020.xlsx", header=5)[
#     ["Operator Name", "Operator Id", "Plant State", "Sector Name"]]
#
# eia_data = eia_data[eia_data["Sector Name"] == "Electric Utility"].drop("Sector Name", axis=1).drop_duplicates().sort_values("Plant State")
#
# eia_data = eia_data[eia_data["Operator Name"] != "State-Fuel Level Increment"]
#
# eia_data.to_csv("Utility Tariffs/utilities_state_mapper.csv", index=False)

### create a database of utility tariffs from the OpenEI Utility Tariff database. Filter these down and one keep a small amount of info for each. The rest of the data can be acquired on the fly later with a call to the openEI API
# rate_db = pd.read_csv("/Users/jhthompson12/Desktop/usurdb.csv")
# rate_db = rate_db[(rate_db["sector"] == "Residential") & (~rate_db["startdate"].isna()) & (rate_db["enddate"].isna())]
#
# rate_db["eiaid"] = rate_db["eiaid"].astype("int")
#
# rate_db = rate_db.drop(["startdate", "enddate", "sector", "source", "sourceparent"], axis=1)
# rate_db = rate_db[["label", "eiaid", "name", "utility", "basicinformationcomments"]]
#
# rate_db.to_csv("Utility Tariffs/tariff_db.csv", index=False)

# Import the list of rates for the user to choose from
rate_db = pd.read_csv("tariff_db.csv")

# send a request to OpenEI API to get the selected rate info. https://openei.org/services/doc/rest/util_rates/
api_url = "https://api.openei.org/utility_rates?params"


def configRateStruct(rate_struct_list):
    rate_struct = pd.DataFrame()
    p = 0
    for period in rate_struct_list:
        t = 0
        for tier in period:
            ind = pd.MultiIndex.from_tuples([(p, t)], names=["Period", "Tier"])
            row = pd.DataFrame(tier, index=ind)
            rate_struct = pd.concat([rate_struct, row])
            t += 1
        p += 1
    return rate_struct


class residentialTariff:
    def __init__(self, rate_identifier):
        req_params = paramets
        req_params["getpage"] = rate_identifier

        # save the response from the API call for troubleshooting
        self.apiResponse = requests.get(api_url, params=req_params)

        # unpack the json goods as a dict for analysis, save for troubleshooting
        self.tariffData = json.loads(self.apiResponse.text)["items"][0]

        # save the tariff name
        self.name = self.tariffData["name"]

        # This saves the link to a page which shows all the information about this tariff
        self.source_page = self.tariffData["uri"]

        # get energy related rate structure, if it exists
        try:
            self.energyComp = self.EnergyComponent(self.tariffData)
        except KeyError as e:
            pass

        # get energy related rate structure, if it exists
        try:
            self.demandComp = self.DemandComponent(self.tariffData)
        except KeyError as e:
            pass

        # get energy related rate structure, if it exists
        try:
            self.fixedComp = self.FixedComponent(self.tariffData)
        except KeyError as e:
            pass

    class EnergyComponent:
        def __init__(self, tariffData):
            self.energyRateStruct = configRateStruct(tariffData["energyratestructure"])
            self.energyWeekdaySched, self.energyWeekendSched = self.energySched(tariffData["energyweekdayschedule"], tariffData["energyweekendschedule"])

        def energySched(self, weekday, weekend):
            weekday_sched = pd.DataFrame()
            weekend_sched = pd.DataFrame()
            for m in range(len(weekday)):
                weekday_sched = pd.concat([weekday_sched, pd.DataFrame(weekday[m], columns=[m]).T])
                weekend_sched = pd.concat([weekend_sched, pd.DataFrame(weekend[m], columns=[m]).T])

            return weekday_sched, weekend_sched

    class DemandComponent:
        def __init__(self, tariffData):
            self.flatDemandRateStruct = configRateStruct(tariffData["flatdemandstructure"])
            self.flatDemandMonthSched = pd.DataFrame(tariffData["flatdemandmonths"],
                                                     columns=["Months"])
            self.flatDemandUnit = tariffData["flatdemandunit"]

    class FixedComponent:
        def __init__(self, tariffData):
            self.fixedChargeUnits = tariffData['fixedchargeunits']
            self.fixedChargeAmount = tariffData['fixedchargefirstmeter']


class year_bill:

    def __init__(self, load_series, tariffId):
        residentialTariffObj = residentialTariff(tariffId)

        self.energyCosts = pd.Series(dtype="object")
        self.demandCosts = pd.Series(dtype="object")
        self.fixedCosts = pd.Series(dtype="object")
        self.Total = 0

        load_df = pd.concat(
            [load_series.rename("load"), pd.Series(load_series.index.month, index=load_series.index).rename("month")],
            axis=1)

        for m in load_df.month.unique():
            month_data = load_df[load_df.month == m].copy()
            month_data["Week End or Day?"] = ["day" if x.weekday() in range(0, 5) else "end" for x in month_data.index]  # month_data.index.apply(lambda x: "day" if x.weekday in range(0,5) else "end")

            # calculate the fixed portion of this months costs, if it exists
            try:
                if residentialTariffObj.fixedComp.fixedChargeUnits == "$/month":
                    monthFixedCost = residentialTariffObj.fixedComp.fixedChargeAmount
                    self.fixedCosts = self.fixedCosts.append(pd.Series([monthFixedCost], index=[m]))
                    self.Total += monthFixedCost
            except:
                self.fixedCosts = np.nan

            # calculate the energy portion of this months costs, if it exists
            try:
                monthEnergyCost = 0
                rate_struct = residentialTariffObj.energyComp.energyRateStruct
                daily_sched = {"day": residentialTariffObj.energyComp.energyWeekdaySched.loc[m - 1]}
                daily_sched["end"] = residentialTariffObj.energyComp.energyWeekendSched.loc[m - 1]

                # If this months rate structure is tiered, rather than time of use
                if len(rate_struct.index.get_level_values(1).unique()) > 1:
                    month_period = daily_sched["day"].unique()[0]
                    month_tot_ene = month_data["load"].sum()

                    for tier, deets in rate_struct.loc[month_period, :].iterrows():
                        if month_tot_ene > deets["max"]:
                            monthEnergyCost += deets["max"] * deets["rate"]
                            last_max = deets["max"]
                        else:
                            monthEnergyCost += (month_tot_ene - last_max) * deets["rate"]


                # if else, it must be a TOU or flat rate which is calculated here
                else:
                    month_data["hour"] = month_data.index.hour
                    monthEnergyCost += month_data.apply(lambda x: x["load"] * rate_struct.loc[daily_sched[x["Week End or Day?"]].loc[x["hour"]], 0]["rate"], axis=1).sum()

                self.energyCosts = self.energyCosts.append(
                    pd.Series([monthEnergyCost], index=[m]))
                self.Total += monthEnergyCost

            except:
                self.energyCosts = np.nan

            # calculate the demand portion of this months bill, if it exists
            try:
                demandPeriod = residentialTariffObj.demandComp.flatDemandMonthSched.loc[m - 1, "Months"]
                monthDemandCost = residentialTariffObj.demandComp.flatDemandRateStruct.loc[demandPeriod, 0]["rate"] * max(month_data["load"])
                self.demandCosts = self.demandCosts.append(
                    pd.Series([monthDemandCost], index=[m]))
                self.Total += monthDemandCost
            except:
                self.demandCosts = np.nan


if __name__ == "__main__":

    ### This was used to create the rate objects
    rates_6 = {}
    for label in rate_db.iloc[4995:]["label"]:
        try:
            rates_6[label] = residentialTariff(label)
        except KeyError as e:
            pass
            # print(label)

    with open('tariff_objects_Jun8_6.pkl', 'wb') as f:
        pickle.dump(rates_6, f)

    files = ["tariff_objects_Jun7.pkl", "tariff_objects_Jun7_2.pkl", "tariff_objects_Jun8_3.pkl", "tariff_objects_Jun8_4.pkl", "tariff_objects_Jun8_5.pkl", "tariff_objects_Jun8_6.pkl"]
    all_tariffs = {}

    for file in files:
        with open(file, 'rb') as f:
            rates_loaded = pickle.load(f)
        for label, obj in rates_loaded.items():
            all_tariffs[label] = obj

    rate_db_available = rate_db[[key in all_tariffs.keys() for key in rate_db["label"].values]]

    rate_db_available.to_pickle("Post Processed Tariff Data/available_rates.pkl")
    with open('Post Processed Tariff Data/rate_objects.pkl', 'wb') as f:
        pickle.dump(all_tariffs, f)

    #
    # ## Load in the already created load objects
    # with open('tariff_objects.pkl', 'rb') as f:
    #     rates_loaded = pickle.load(f)
    #
    # # just the ga power rates
    # gapower_rates = rate_db[rate_db["eiaid"] == 7140]
    #
    # print(gapower_rates)

    ### import a load
    # load = pd.read_pickle(git_data_url + "residential_load_pickles/BASE/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3_BASE.pkl")
    #
    # electricity_costs = year_bill(load, tariffId="5d5584155457a3391535a086")
