from strands import Agent, tool
from strands.memory import MemoryManager
from strands.vended_tools import file_editor
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

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

project = TestMemoryStore(name="notes", path="./notes.json")

agent = Agent(memory_manager=MemoryManager(stores=[project], add_tool_config=True), tools=[letter_counter, file_editor])
# Bedrock is the default, so no model object is needed.
agent('How many letter R\'s are in the word "orange"? Write the answer to answer.txt.')

