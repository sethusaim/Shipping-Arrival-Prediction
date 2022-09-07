import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.logger import App_Logger
from utils.preprocess_utils import Preprocess_Utils
from utils.read_params import get_log_dic, read_params


class Preprocessor:
    """
    Description :   This class shall  be used to clean and transform the data before training.
    Version     :   1.2
    
    Revisions   :   moved setup to cloud
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.config = read_params()

        self.train_input_csv_file = (
            self.config["train_input_dir"]
            + "/"
            + self.config["export_csv_file"]["train"]
        )

        self.cols_to_be_one_hot_encoded = self.config["preprocess_cols"][
            "one_hot_encode"
        ]

        self.cols_to_be_ordinally_encoded = self.config["preprocess_cols"][
            "ordinal_encode"
        ]

        self.columns_to_drop = self.config["preprocess_cols"]["remove"]

        self.log_writer = App_Logger()

        self.preprocess_utils = Preprocess_Utils(self.log_file)

        self.st = StandardScaler()

    def apply_one_hot_encoding(self, data):
        """
        Method Name :   apply_one_hot_encoding
        Description :   This method applies one hot encoding to selected columns 
        
        Output      :   A pandas dataframe after applying one hot encoding
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_one_hot_encoding.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Applying one hot encoding to the dataframe for the particular columns",
                **log_dic
            )

            df_train = self.preprocess_utils.one_hot_encoding(
                data, self.cols_to_be_one_hot_encoded
            )

            self.log_writer.log("Converted dataframe to csv file", **log_dic)

            self.log_writer.log(
                "Applied one hot encoding for the particular columns", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return df_train

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_ordinal_encoding(self, data):
        """
        Method Name :   apply_ordinal_encoding
        Description :   This method applies ordinal encoding to selected columns 
        
        Output      :   A pandas dataframe after applying ordinal encoding
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_ordinal_encoding.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Applying ordinal encoding to the dataframe for the particular columns",
                **log_dic
            )

            df_train = self.preprocess_utils.ordinal_encoding(
                data, self.cols_to_be_ordinally_encoded
            )

            self.log_writer.log("Converted dataframe to csv file", **log_dic)

            self.log_writer.log(
                "Applied one hot encoding for the particular columns", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return df_train

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def remove_columns(self, data):
        """
        Method Name :   apply_ordinal_encoding
        Description :   This method applies ordinal encoding to selected columns 
        
        Output      :   A pandas dataframe after applying ordinal encoding
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.remove_columns.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Dropping selected columns from dataframe", **log_dic)

            data.drop(self.columns_to_drop, axis=1, inplace=True)

            self.log_writer.log("Dropped selected columns from dataframe", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def separate_label_feature(self, data, label_column_name):
        """
        Method Name :   separate_label_feature
        Description :   This method separates the label column from the dataframe
        
        Output      :   Two pandas dataframe are returned after separating the label column
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.separate_label_feature.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Separating label feature from the dataframe", **log_dic
            )

            X_train = data.drop(label_column_name, axis=1)

            Y_train = data[label_column_name]

            self.log_writer.log("Separated label feature from the dataframe", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return X_train, Y_train

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def apply_standard_scaler(self, data):
        """
        Method Name :   apply_standard_scaler
        Description :   This method applies standard scaling to the dataframe 
        
        Output      :   A pandas dataframe after applying standard scaling
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.apply_standard_scaler.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Applying standard scaling on the dataframe", **log_dic)

            df_train_standardized = self.st.fit_transform(data)

            self.log_writer.log("Applied standard scaling on the dataframe", **log_dic)

            df_train_final = pd.DataFrame(df_train_standardized, columns=data.columns)

            self.log_writer.log(
                "Created dataframe after applying standard scaling", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return df_train_final

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
