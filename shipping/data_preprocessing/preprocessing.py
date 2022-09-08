import numpy as np
from pandas import DataFrame
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from utils.logger import App_Logger
from utils.main_utils import Main_Utils
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

        self.null_values_file = self.config["null_values_csv_file"]

        self.target_col = self.config["target_col"]

        self.cols_to_be_one_hot_encoded = self.config["preprocess_cols"][
            "one_hot_encode"
        ]

        self.cols_to_be_ordinally_encoded = self.config["preprocess_cols"][
            "ordinal_encode"
        ]

        self.columns_to_drop = self.config["preprocess_cols"]["remove"]

        self.artifact_folder = self.config["dir"]["artifacts"]

        self.knn_params = self.config["knn_imputer"]

        self.log_writer = App_Logger()

        self.utils = Main_Utils()

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
                **log_dic,
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
                **log_dic,
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

            df_train_final = DataFrame(df_train_standardized, columns=data.columns)

            self.log_writer.log(
                "Created dataframe after applying standard scaling", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return df_train_final

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def is_null_present(self, data):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.is_null_present.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        self.null_present = False

        self.cols_with_missing_values = []

        self.cols = data.columns

        try:
            self.null_counts = data.isna().sum()

            self.log_writer.log(f"Null values count is : {self.null_counts}", **log_dic)

            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True

                    self.cols_with_missing_values.append(self.cols[i])

            self.log_writer.log("created cols with missing values", **log_dic)

            self.utils.create_directory(self.artifact_folder, self.log_file)

            if self.null_present:
                self.log_writer.log(
                    "null values were found the columns...preparing dataframe with null values",
                    **log_dic,
                )

                self.dataframe_with_null = DataFrame()

                self.dataframe_with_null["columns"] = data.columns

                self.dataframe_with_null["missing values count"] = np.asarray(
                    data.isna().sum()
                )

                self.log_writer.log("Created dataframe with null values", **log_dic)

                self.dataframe_with_null.to_csv(
                    self.null_values_file, index=None, header=True
                )

                self.log_writer.log(
                    "Converted null values dataframe to csv file", **log_dic
                )

            else:
                self.log_writer.log(
                    "No null values are present in cols. Skipped the creation of dataframe",
                    **log_dic,
                )

            self.log_writer.start_log("exit", **log_dic)

            return self.null_present

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def impute_missing_values(self, data):
        """
        Method Name :   impute_missing_values
        Description :   This method replaces all the missing values in the dataframe using mean values of the column.
        
        Output      :   A dataframe which has all the missing values imputed.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.impute_missing_values.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.data = data

            imputer = KNNImputer(missing_values=np.nan, **self.knn_params)

            self.log_writer.log(f"Initialized {imputer.__class__.__name__}", **log_dic)

            self.new_array = imputer.fit_transform(self.data)

            self.new_data = DataFrame(data=self.new_array, columns=self.data.columns)

            self.log_writer.log("Created new dataframe with imputed values", **log_dic)

            self.log_writer.log("Imputing missing values Successful", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return self.new_data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def remove_target_column(self, data):
        """
        Method Name :   remove_target_column
        Description :   This method removes the target column in the dataframe
        
        Output      :   A target column has been removed from the dataframe
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.remove_target_column.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Removing target column from the dataframe", **log_dic)

            new_order = list(data.columns)

            new_order.remove(self.target_col)

            self.log_writer.log("Removed target column from the dataframe", **log_dic)

            data = data.reindex(columns=new_order)

            self.log_writer.log(
                "Reindex the dataframe based on the new order", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return data

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
