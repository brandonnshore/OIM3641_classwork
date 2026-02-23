"""
Session 7: Class Structure, Encapsulation, and Abstraction.

This module provides a class to validate user-submitted text against 
length and security requirements.
"""

import string


class InputValidator:
    """
    A class used to represent an input validation engine.

    Attributes:
        text (str): The sanitized (stripped) version of the input text.
    """

    def __init__(self, text):
        """
        Initializes the validator and performs basic encapsulation.

        Args:
            text (str): The raw string to be validated.
        """
        # TODO: Store the stripped version of 'text' in self.text
        pass

    def is_long_enough(self, min_chars=20):
        """
        Internal logic to verify the length of the string.

        Args:
            min_chars (int): The threshold for a valid string. Defaults to 20.

        Returns:
            bool: True if length is >= min_chars, False otherwise.
        """
        pass

    def is_safe(self):
        """
        Internal logic to check for dangerous SQL keywords and punctuation.

        Security Criteria:
        1. No SQL keywords: SELECT, DELETE, INSERT, UPDATE, DROP, --, ;
        2. No characters from string.punctuation.

        Returns:
            bool: True if no dangerous elements are found, False otherwise.
        """
        pass

    def validate_all(self):
        """
        The Public Interface: Abstracts the internal validation logic.

        This method coordinates the execution of is_long_enough and is_safe.

        Returns:
            tuple: (bool, str) 
                   - The bool indicates success/failure.
                   - The str provides the specific success or error message.
        """
        pass

    if __name__ == "__main__":
        # add code to test your class below
        pass