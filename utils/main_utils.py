from json import load
from os import makedirs
from os.path import isdir

from utils.logger import App_Logger
from utils.read_params import get_log_dic, read_params


class Main_Utils:
    def __init__(self):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.good_data_dir = self.config["data"]["train"]["good_data_dir"]

        self.bad_data_dir = self.config["data"]["train"]["bad_data_dir"]

    def read_json(self, file, log_file):
        """
        Method Name :   read_json
        Description :   This method reads the json data from the file and returns the content

        Output      :   Json data is read from the file and file content is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.read_json.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(f"Reading json data from {file} file", **log_dic)

            with open(file, "r") as f:
                dic = load(f)

            self.log_writer.log(f"Read the json data from the {file} file", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

            return dic

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def read_text(self, file, log_file):
        """
        Method Name :   read_text
        Description :   This method reads the text data from the file and returns the content

        Output      :   Text data is read from the file and file content is returned
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.read_text.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(f"Reading text data from {file} file", **log_dic)

            with open(file, "r") as f:
                content = f.readline()

            self.log_writer.log(
                f"Read the text data content from {file} file", **log_dic
            )

            self.log_writer.start_log("exit", **log_dic)

            return content

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def create_directory(self, folder, log_file, exist_ok=True):
        """
        Method Name :   create_directory
        Description :   This method creates folders based on the folder name

        Output      :   Folder is created in the particular directory
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__, self.create_directory.__name__, __file__, log_file
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log(
                f"Creating folder with folder name as {folder}", **log_dic
            )

            if not isdir(folder):
                makedirs(folder, exist_ok=exist_ok)

            self.log_writer.log("Folder created with {folder} folder name", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)

    def create_dirs_for_good_bad_data(self, log_file):
        """
        Method Name :   create_dirs_for_good_bad_data
        Description :   This method creates good and bad data folders 

        Output      :   Good and bad data folder are created
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        log_dic = get_log_dic(
            self.__class__.__name__,
            self.create_dirs_for_good_bad_data.__name__,
            __file__,
            log_file,
        )

        self.log_writer.start_log("start", **log_dic)

        try:
            self.log_writer.log("Creating folders for good and bad data", **log_dic)

            self.create_directory(self.good_data_dir, log_file)

            self.create_directory(self.bad_data_dir, log_file)

            self.log_writer.log("Created folders for good and bad data", **log_dic)

            self.log_writer.start_log("exit", **log_dic)

        except Exception as e:
            self.log_writer.exception_log(e, **log_dic)