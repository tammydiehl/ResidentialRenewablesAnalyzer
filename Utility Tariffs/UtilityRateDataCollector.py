import pandas as pd
import requests
import json

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
rate_db = pd.read_csv("Utility Tariffs/tariff_db.csv")

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


if __name__ == "__main__":

    rates = {}
    for label in rate_db["label"]:    # [rate_db["eiaid"] == 7140]
        try:
            rates[label] = residentialTariff(label)
        except KeyError as e:
            pass
            # print(label)
