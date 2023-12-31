import sys

def error_message_detail(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class ShippingException(Exception):
    def __init__(self, error_message, error_detail=None):
        if error_detail is None:
            error_detail = sys.exc_info()
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
