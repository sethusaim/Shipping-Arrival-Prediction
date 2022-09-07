import numpy as np
import pandas as pd

from utils.logger import App_Logger
from utils.read_params import get_log_dic


class Data_Transform_Utils:
    def __init__(self, log_file):
        self.log_writer = App_Logger()

        self.log_file = log_file

    def fill_mean(self, df):
        log_dic = get_log_dic(
            self.__class__.__name__, self.fill_mean.__name__, __file__, self.log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Filling the dataframe with mean values", **log_dic)

            values, counts = [], []

            for i, j in df.value_counts().iteritems():
                values.append(i)
                counts.append(j)

            weighted_avg_artist_reputation = np.average(values, weights=counts)

            df.fillna(weighted_avg_artist_reputation, inplace=True)

            self.log_writer.log("Filled the dataframe with mean values", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return df

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def fill_not_available(self, df):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.fill_not_available.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Starting filling dataframe with mode values", **log_dic
            )

            df.fillna(df.mode()[0], inplace=True)

            self.log_writer.log("Filled the dataframe cols with mode", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return df

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def clean_weight(self, df):
        log_dic = get_log_dic(
            self.__class__.__name__, self.clean_weight.__name__, __file__, self.log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Cleaning weight col in the dataframe", **log_dic)

            converted_list_1 = []

            for i in df:
                converted_list_1.append(round(float(i), 2))

            self.log_writer.log("Cleaned weight column in the dataframe", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return converted_list_1

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def change_date_time(self, df, i):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.change_date_time.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Changed datetime for cols in the dataframe", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return pd.to_datetime(df[i])

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def clean_date(self, df):
        log_dic = get_log_dic(
            self.__class__.__name__, self.clean_date.__name__, __file__, self.log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Started cleaning date for the dataframe", **log_dic)

            converted_list_1 = []

            date_diff = df["Scheduled Date"] - df["Delivery Date"]

            for i in date_diff:
                converted_list_1.append(str(i).split()[0])

            self.log_writer.log("Cleaning date for the dataframe", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return converted_list_1

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def clean_customer_location(self, df):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.clean_customer_location.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Cleaning customer location data for dataframe", **log_dic
            )

            converted_list_1 = []

            for i in df:
                converted_list_1.append(i.split()[-2])

            self.log_writer.log(
                "Cleaned customer location data for dataframe", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return converted_list_1

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
