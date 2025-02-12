"""
Flare JSON Updater

This script allows the user to update the 'Flares.json' file by retreiving
the necessary file from the NASA Goddard Space Flight Center. Then, the JSON file
(ActiveRegionEventTimes.json) that contains all the AR # to event time entries is
updated to reflect the contents of the updated Flares.json file.

This is important for keeping flare information at specific dates up-to-date
so that CalculateProxies can retrieve accurate information from the JSON file
while training.

Developers
----------
Ian Fernandes: fernandesi2244@gmail.com

NASA JSC, Space Medicine Operations Division (SD2), Space Radiation Analysis Group (SRAG)

Last Updated 2020
"""

import datetime
import json
import os
import pathlib

import wget

def main():
    """
    Updates the DONKI flare database.
    """
    
    # NOTE: This file should be run before training so that the JSON data is up-to-date and so that results will be accurate.
    # Also, keep in mind that if the file update fails, the URLs for the two JSON files may have changed. In this case, locate
    # the equivalent links on the NASA DONKI website and update the URLs below.

    # Download observed flare data JSON; if the file already exists, update it with the new version.
    fileName = '../InputData/Flares.json'
    if os.path.exists(fileName):
        os.remove(fileName)
    url = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate=2010-06-12' # First flare in DONKI database on 6/12/2010
    print("---Not sure what the printout below means, but it doesn't seem to affect the download.---\nSTART: ")

    retrievedFile = wget.download(url, fileName)
    print('\nEND\n')
    print('Observed flare JSON file downloaded.')

if __name__ == '__main__':
    main()