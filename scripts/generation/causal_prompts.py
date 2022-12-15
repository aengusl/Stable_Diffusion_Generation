from statistics import correlation
from turtle import pos
from assets import OBJECTS, LOCATIONS, TIMES_OF_DAY, WEATHER, POSITIONS
import random

# Things I could add:
# 1. input for environment, and strength of correlations in environment
# 2. input for which variables to exclude, more logic than correlation
# 3. output the variables in the prompt

def get_prompt():

    object = random.choice(OBJECTS)
    position = random.choice(POSITIONS)
    location = random.choice(LOCATIONS)
    time_of_day = random.choice(TIMES_OF_DAY)
    weather = random.choice(WEATHER)

    an = "an" if object[0] in "aeiou" else "a"
    body = f" {object} in the {position} of the image, {location}, {time_of_day}, {weather}, highly detailed, with cinematic lighting"
    prompt = an + body

    return prompt

if __name__ == "__main__":
    print(get_prompt())