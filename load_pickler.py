import os
import pandas as pd
import datetime


mydateparser = lambda x: datetime.datetime.strptime("2020-%s %s:00:00" % (x.split()[0], int(x.split()[-1].split(":")[0])-1), "%Y-%m/%d %H:%M:%S")

if __name__ == "__main__":
    # import all load csvs, condense to one column, pickle the data
    load_files_avail = pd.DataFrame()
    for root, dirs, files in os.walk("residential_load_data"):
        path = root.split(os.sep)
        for file in files:
            info_dict = {}
            if file[:3] == "USA":
                try:
                    split_name = file.split("_")
                    info_dict["State"] = split_name[1]
                    info_dict["Locale"] = " ".join(split_name[2].split(".")[:-1])
                    info_dict["Load Level"] = split_name[-1][:-4]
                    info_dict["File Dir"] = "residential_load_pickles/" + root.split("/")[-1] + "/" + file[:-3] + "pkl"
                    load_files_avail = pd.concat([load_files_avail, pd.DataFrame(info_dict, index=[0])], ignore_index=True)

                    pd.read_csv(root + "/" + file, date_parser=mydateparser, parse_dates=[0]).set_index(
                        'Date/Time').sum(axis=1).to_pickle(info_dict["File Dir"])

                except Exception as e:
                    print("issue with " + file)

    load_files_avail.to_pickle("available_loads_summary.pkl")

    # data = pd.read_csv("BASE/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3_BASE.csv", date_parser=mydateparser, parse_dates=[0]).set_index('Date/Time')
