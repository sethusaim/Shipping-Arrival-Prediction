from os import listdir

from numpy import log1p
from pandas import read_csv

from utils.data_transform_utils import Data_Transform_Utils
from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Data_Transform_Train:
    def __init__(self):
        self.config = read_params()

        self.train_data_transform_log = self.config["log"]["train_data_transform"]

        self.good_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.mean_to_be_filled = ["Artist Reputation", "Height", "Width", "Weight"]

        self.not_available_to_be_filled = ["Transport", "Material", "Remote Location"]

        self.data_transform_utils = Data_Transform_Utils(self.train_data_transform_log)

        self.log_writer = App_Logger()

    def apply_log1p_transform(self):
        """
        Method Name :   apply_log1p_transform
        Description :   This method applies the log1p transformation on the dataframe and returns it 
        
        Output      :   A dataframe is returned after applying log1p transformation
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_log1p_transform.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Applying log1p transformation on training data", **log_dic
            )

            for file in listdir(self.good_data_dir):
                fname = self.good_data_dir + "/" + file

                train_data = read_csv(fname)

                cost = train_data["Cost"]

                train_data["Cost"] = log1p(abs(cost))

                self.log_writer.log(
                    f"Applied log1p transformation for {fname} filename", **log_dic
                )

                train_data.to_csv(fname, index=None, header=True)

                self.log_writer.log(
                    f"Converted dataframe to csv with filename as {fname}", **log_dic
                )

            self.log_writer.log(
                "Applied log1p transformation on training data", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_clean_customer_location_transformation(self):
        """
        Method Name :   apply_clean_customer_location_transformation
        Description :   This method applies the cleans the customer location data and returns the dataframe
        
        Output      :   A dataframe is returned after cleaning the customer location
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_clean_customer_location_transformation.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Cleaning customer location data", **log_dic)

            for file in listdir(self.good_data_dir):
                fname = self.good_data_dir + "/" + file

                train_data = read_csv(fname)

                train_data[
                    "Customer Location"
                ] = self.data_transform_utils.clean_customer_location(
                    train_data["Customer Location"]
                )

                self.log_writer.log(
                    f"Cleaned customer location data for {fname} filename", **log_dic
                )

                train_data.to_csv(fname, index=None, header=True)

                self.log_writer.log(
                    f"Converted dataframe to {fname} filename", **log_dic
                )

            self.log_writer.log("Cleaned customer location data", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_date_time_transformation(self):
        """
        Method Name :   apply_date_time_transformation
        Description :   This method changes the datetime to required format
        
        Output      :   A dataframe is returned after changing the datetime format
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_date_time_transformation.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Changing the datetime format in the dataframe", **log_dic
            )

            for file in listdir(self.good_data_dir):
                fname = self.good_data_dir + "/" + file

                train_data = read_csv(fname)

                cols_to_change_date = ["Scheduled Date", "Delivery Date"]

                for i in cols_to_change_date:
                    train_data[i] = self.data_transform_utils.change_date_time(
                        train_data, i
                    )

                train_data["date_diff"] = self.data_transform_utils.clean_date(
                    train_data
                )

                self.log_writer.log(f"Cleaned date for {fname} filename", **log_dic)

                train_data["date_diff"] = train_data["date_diff"].astype("int")

                train_data.to_csv(fname, index=None, header=True)

                self.log_writer.log(
                    f"Converted dataframe to csv for {fname} filename", **log_dic
                )

            self.log_writer.log(
                "Changed the datetime format in the dataframe", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_clean_weight_transformation(self):
        """
        Method Name :   apply_clean_weight_transformation
        Description :   This method cleans the weight column in the dataframe
        
        Output      :   A dataframe is returned after cleaning the weight column
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_clean_weight_transformation.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Cleaning the weight column in the dataframe", **log_dic
            )

            for file in listdir(self.good_data_dir):
                fname = self.good_data_dir + "/" + file

                train_data = read_csv(fname)

                train_data["Weight"] = self.data_transform_utils.clean_weight(
                    train_data["Weight"]
                )

                train_data.to_csv(fname, index=None, header=True)

            self.log_writer.log("Cleaned the weight column in the dataframe", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_mean_transformation(self):
        """
        Method Name :   apply_mean_transformation
        Description :   This method fills the dataframe with mean in the cols to be filled with mean
        
        Output      :   A dataframe is returned after applying mean in the selected cols
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_mean_transformation.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Applying mean in selected cols", **log_dic)

            for file in listdir(self.good_data_dir):
                fname = self.good_data_dir + "/" + file

                train_data = read_csv(fname)

                for i in self.mean_to_be_filled:
                    mean_train_df = self.data_transform_utils.fill_mean(train_data[i])

                    mean_train_df.to_csv(fname, index=None, header=True)

            self.log_writer.log("Applied mean in selected cols", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_fill_to_not_avaiable_cols(self):
        """
        Method Name :   apply_fill_to_not_avaiable_cols
        Description :   This method fills the dataframe with mode to other cols
        
        Output      :   A dataframe is returned after applying mode to other cols
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_fill_to_not_avaiable_cols.__name__,
            __file__,
            self.train_data_transform_log,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Applying mode to other cols", **log_dic)

            for file in listdir(self.good_data_dir):
                fname = self.good_data_dir + "/" + file

                train_data = read_csv(fname)

                for i in self.not_available_to_be_filled:
                    fill_not_available_df = self.data_transform_utils.fill_not_available(
                        train_data[i]
                    )

                    fill_not_available_df.to_csv(fname, index=None, header=True)

            self.log_writer.log("Applied mode to other cols", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
