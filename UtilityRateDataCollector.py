import pandas as pd
import requests


### Use this plant data to get a mapping of electric utilities to EIA IDs and the states that the utility is in
# eia_data = pd.read_excel("/Users/jhthompson12/Desktop/EIA923_Schedules_2_3_4_5_M_02_2020_20APR2020.xlsx", header=5)[
#     ["Operator Name", "Operator Id", "Plant State", "Sector Name"]]
#
# eia_data = eia_data[eia_data["Sector Name"] == "Electric Utility"].drop("Sector Name", axis=1).drop_duplicates().sort_values("Plant State")
#
# eia_data = eia_data[eia_data["Operator Name"] != "State-Fuel Level Increment"]
#
# eia_data.to_csv("Utility Tariffs/utilities_state_mapper.csv", index=False)

rate_db = pd.read_csv("/Users/jhthompson12/Desktop/usurdb.csv")
rate_db = rate_db[(rate_db["sector"] == "Residential") & (~rate_db["startdate"].isna()) & (rate_db["enddate"].isna())]

rate_db["eiaid"] = rate_db["eiaid"].astype("int")

rate_db = rate_db.drop(["startdate", "enddate", "sector", "source", "sourceparent"], axis=1)
rate_db = rate_db[["label", "eiaid", "name", "utility", "basicinformationcomments"]]

rate_db.to_csv("Utility Tariffs/tariff_db.csv", index=False)

# send a request to OpenEI to get the selected rate info
api_url = "https://api.openei.org/utility_rates?params"

paramets = {
    "version": "latest",
    "format": "csv",
    "api_key": "snVIu7FREbmucvgeZjPgsumRjdCCqQtwHd0QJsmb",
    "getpage": "5d5582305457a3e01135a086"
}

tariff_response = requests.get(api_url, params=paramets)

f = open("Utility Tariffs/utility_response.csv", "w")
f.write(tariff_response.text)
f.close()

selected_tariff = pd.read_csv("Utility Tariffs/utility_response.csv").dropna(axis=1)
