class InvalidDataError(Exception):
    """Exception raised when the current data sample is not valid."""

    def __init__(self):
        super(InvalidDataError, self).__init__()
