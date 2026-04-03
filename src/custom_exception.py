# Define a custom exception class for the project.
class ProjectException(Exception):
    # Initialize the custom exception with an error message.
    def __init__(self, message: str):
        # Call the parent Exception class constructor.
        super().__init__(message)

        # Store the message in an instance variable.
        self.message = message