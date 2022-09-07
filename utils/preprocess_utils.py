from category_encoders import OneHotEncoder, OrdinalEncoder

from utils.logger import App_Logger
from utils.read_params import get_log_dic


class Preprocess_Utils:
    def __init__(self, log_file):
        self.log_file = log_file

        self.log_writer = App_Logger()

    def one_hot_encoding(self, data, column):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.one_hot_encoding.__name__,
            __file__,
            self.log_file,
        )

        try:
            self.log_writer.log(
                "Applying one hot encoder to selected columns", **log_dic
            )

            one_hot_encoder = OneHotEncoder(
                cols=column, return_df=True, use_cat_names=True
            )

            data_final = one_hot_encoder.fit_transform(data)

            self.log_writer.log("Applied one hot encoder to columns", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return data_final

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def ordinal_encoding(self, data, column):
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.ordinal_encoding.__name__,
            __file__,
            self.log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                "Applying ordinal encoding to dataframe for particular columns",
                **log_dic
            )

            ordinal_encoder = OrdinalEncoder(cols=column, return_df=True)

            df_final = ordinal_encoder.fit_transform(data)

            self.log_writer.log(
                "Applied ordinal encoding to dataframe for particular cols", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return df_final

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)
