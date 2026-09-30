from strands import Agent, tool
from strands.vended_tools import file_editor

@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")

    return word.lower().count(letter.lower())

# Bedrock is the default, so no model object is needed.
agent = Agent(tools=[letter_counter, file_editor])
agent('How many letter R\'s are in the word "strawberry"? Write the answer to answer.txt.')

