"""
Handles the downloading and parsing of NOAA Solar Region Summary files.

This abstraction is important because any programs needing information from 
an SRS file will not have to worry about the complications of file downloads
and data parsing.

Developers
----------
Ian Fernandes: fernandesi2244@gmail.com
Tilaye Tadesse: Tilaye.T.Asfaw@nasa.gov

NASA JSC, Space Medicine Operations Division (SD2), Space Radiation Analysis Group (SRAG)

Last Updated 2020
"""

import os
import pathlib
import re
import shutil
import sys
import tarfile
from datetime import datetime, timedelta
import wget

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, rootDir)

from Logger import Logger
from SRS_AR import SRS_AR


class SRSHandler(object):
    """
    Represents an object that can handle all operations for an SRS file at a specific date that 
    corresponds to the HARP being examined in CalculateProxies.
    """

    # SRS_DIR_PATH = os.path.join(rootDir, 'Input Data', 'Test Data', 'NOAA', 'SRS') # NOTE: Automatically changes to Operational Data folder in downloadSRS() if necessary
    SRS_DIR_PATH = os.path.join(os.sep+'share', 'development', 'data', 'drms', 'MagPy_Shared_Data', 'SRSFiles')

    # logger = Logger()
    logger = Logger('VolumeParameterizationLog.txt')

    def __init__(self, HARPDate: datetime):
        """
        Initialize the SRSHandler object with a HARP date.

        :param HARPDate: the date of the associated SHARP blob
        """

        self.HARPDate = HARPDate
        self.file = None
        self.sunspotLine = None
        self.triesRemaining = 14 # Approx. number of days for AR to get from East limb to West limb @ equator (ignore differential rotation)
    
    def downloadSRS(self, NRT=False):
        """
        Downloads relevant NOAA Solar Region Summary file to File Cache directory.

        Note: SRS files can only be retrieved for years in interval [1996, PRESENT).
        TODO: Add troubleshooting for failed file retrievals; potential corruption checks.
        TODO: Return boolean representing whether or not download worked.

        :param NRT: whether or not the program is in NRT operational use
        """

        # Get components of date keeping in mind that the SRS with day x represents the previous day (x-1)
        year, month, day = (self.HARPDate.year, self.HARPDate.month, self.HARPDate.day)

        print(f'Attempting download of relevant SRS file with date {month}/{day}/{year}.')
        self.triesRemaining -= 1
        
        if NRT:
            SRSHandler.SRS_DIR_PATH = os.path.join(rootDir, 'Input Data', 'Operational Data', 'NOAA', 'SRS')
        
        # Final file name of the SRS file to be downloaded
        if NRT:
            file_name = os.path.join(SRSHandler.SRS_DIR_PATH, f"{year}_{month:02d}_{day:02d}_SRS.txt")
        else:
            file_name = os.path.join(SRSHandler.SRS_DIR_PATH, str(year), f"{year}_{month:02d}_{day:02d}_SRS.txt")

        if NRT:
            try:
                if os.path.isfile(file_name):
                    print('SRS file already exists')
                    self.file = file_name
                    return

                # After verifying the necessary SRS file does not exist, keep the old one (if it exists) and try downloading the requested one.
                oldSRSFile = os.listdir(SRSHandler.SRS_DIR_PATH)[0] # Most recent SRS file downloaded (only keeps one at a time for redundancy)
                self.file = os.path.join(SRSHandler.SRS_DIR_PATH, oldSRSFile)
            except:
                pass # No SRS files downloaded yet. Continue with the download of the new file.

        else:
            # Create directory for year if necessary
            year_dir_to_make = os.path.join(SRSHandler.SRS_DIR_PATH, str(year))
            if not os.path.isdir(year_dir_to_make):
                os.makedirs(year_dir_to_make)

        try:
            '''
            Use try-except instead of checking years because we don't currently know the exact conditions and times for which NOAA generates a tar file for a year.
            There are many factors to account for, so we just try the 'wget SRS directly' method first, and if that doesn't work, we try the 'wget tar -> extract SRS' method. 
            '''
            try:
                # Usually: <Current year> file retrieval = wget SRS directly
                url = f'ftp://ftp.swpc.noaa.gov/pub/warehouse/{year}/SRS/{year}{month:02d}{day:02d}SRS.txt'
                if not os.path.isfile(file_name):
                    wget.download(url, file_name) # Should throw an exception if SRS year is not current year (therefore, we would need to use the tar file method)
                    print('\nSRS file downloaded.')
                else:
                    print('SRS file already exists.')
                
                if os.path.exists(file_name): # SRS file download confirmed
                    if self.file is not None:
                        os.remove(self.file)
                    self.file = file_name
                else:
                    SRSHandler.logger.log(f'SRS file was supposed to exist but does not. Attempted wget download URL: {url}', 'HIGH')
                    raise Exception
            except Exception as inner_exception:
                print(f'Could not wget the SRS file directly. Error: {repr(inner_exception)}. Attempting tar file retrieval method...')
                # Usually: [1996, end of year before current year] file retrieval = wget tar, extract SRS

                # Make TAR dir if necessary
                url = f'ftp://ftp.swpc.noaa.gov/pub/warehouse/{year}/{year}_SRS.tar.gz'

                if NRT:
                    tar_dir_to_make = SRSHandler.SRS_DIR_PATH
                else:
                    tar_dir_to_make = os.path.join(SRSHandler.SRS_DIR_PATH, str(year), 'TAR')
                    if not os.path.isdir(tar_dir_to_make):
                        os.makedirs(tar_dir_to_make)

                # Retrieve TAR file
                file_name = os.path.join(tar_dir_to_make, f"{year}_tar.gz")
                if not os.path.isfile(file_name):
                    wget.download(url, file_name)
                    print('\nTAR file downloaded.')
                else:
                    print('TAR file already exists.')

                # Make sure TAR file was downloaded and can be read by tarfile package
                if not tarfile.is_tarfile(file_name):
                    # TODO: Extra troubleshooting for if it was downloaded but not readable
                    # Perhaps delete existing file and then throw exception
                    SRSHandler.logger.log(f'The downloaded SRS TAR file either does not exist or is not readable. File path: {file_name}', 'HIGH')
                    raise Exception

                # Extract desired SRS file from TAR file
                file_to_extract = os.path.join(f'{year}_SRS/{year}{month:02d}{day:02d}SRS.txt')
                # if year == 2010:
                #     file_to_extract = os.path.join(f'{year}_SRS/{year}{month:02d}{day:02d}SRS.txt')
                if NRT:
                    new_file_path = os.path.join(SRSHandler.SRS_DIR_PATH, f'{year}_{month:02d}_{day:02d}_SRS.txt')
                else:
                    new_file_path = os.path.join(SRSHandler.SRS_DIR_PATH, str(year), f'{year}_{month:02d}_{day:02d}_SRS.txt')
                if not os.path.exists(new_file_path):
                    my_tar = tarfile.open(file_name)
                    my_tar.extract(file_to_extract, tar_dir_to_make) # Should throw an error if an SRS file just doesn't exist for that day (therefore, we should try to get the previous day's SRS file)
                    my_tar.close()
                    print('SRS file downloaded.')

                    if NRT:
                        os.remove(file_name) # Only want SRS files in operational folder (no TARs)
                        if os.path.exists(os.path.join(SRSHandler.SRS_DIR_PATH, f'{year}_SRS')):
                            shutil.rmtree(os.path.join(SRSHandler.SRS_DIR_PATH, f'{year}_SRS')) # Also delete the SRS folder that was created when extracting from the TAR file

                    # Rename extracted SRS file to default format (since specifying a custom file name while extracting is not an option)
                    old_file_path = os.path.join(tar_dir_to_make, file_to_extract)
                    os.rename(old_file_path, new_file_path)
                else:
                    print('SRS file already exists.')

                if os.path.exists(new_file_path): # SRS file download confirmed
                    if self.file is not None:
                        os.remove(self.file)
                    self.file = new_file_path
                else:
                    # TODO: Troubleshoot this (perhaps clean up/delete tar files before throwing exception)
                    SRSHandler.logger.log(f'SRS file was supposed to exist but does not. Attempted SRS extraction from TAR: {file_to_extract}', 'HIGH')
                    raise Exception
        except Exception as outer_exception:
            # TODO: Look at Lockheed database before trying again.
            print('SRS download failed; see below for details.')
            print('\t', repr(outer_exception))
            SRSHandler.logger.log(f'Exception encountered in SRS file download (date: {month}/{day}/{year}): {repr(outer_exception)}. {self.triesRemaining} tries remaining.', 'HIGH')

            # Clean up possible folder that was created if in NRT
            if NRT and os.path.exists(os.path.join(SRSHandler.SRS_DIR_PATH, f'{year}_SRS')):
                shutil.rmtree(os.path.join(SRSHandler.SRS_DIR_PATH, f'{year}_SRS')) # SRS folder that's created when extracting from the TAR file

            if self.triesRemaining > 0:
                print('Decrementing day by one and retrying...')
                self.HARPDate = self.getDateOfLastDay()
                self.downloadSRS(NRT)
            else:
                print('All 14 tries for downloading the SRS file have been used. Moving on with program execution...') # Notice that self.file will be None
    
    def getARList(self, NRT: bool=False) -> list:
        """
        Gets the list of active regions (and plages) within the SRS file.

        NOTE: These do not include the ARs that are due to return within x days.

        :param NRT: whether or not NRT operational mode is in use; if so, will retrieve the recently downloaded SRS file

        :returns: a list of SRS_AR objects that represents each AR (and plage) within the SRS file.
        """

        # Currently, we increment the day by one for testing purposes because each SRS file covers the day before.
        # In NRT, we will only have access to the day before, so we will just have to estimate the AR locations after a day of movement.
        year, month, day = (self.HARPDate.year, self.HARPDate.month, self.HARPDate.day)

        if NRT:
            try:
                fileToGet = os.listdir(os.path.join(rootDir, 'Input Data', 'Operational Data', 'NOAA', 'SRS'))[0] # Assumes only one SRS file in folder at a time
                fileToGet = os.path.join(rootDir, 'Input Data', 'Operational Data', 'NOAA', 'SRS', fileToGet)
            except Exception as e:
                SRSHandler.logger.log(f'SRS file did not exist in operational folder when trying to retrieve AR list. HARP date: {self.HARPDate}', 'HIGH')
                print("SRS File doesn't exist; aborting...")
                return list()
        else:
            fileToGet = self.file
            if fileToGet is None:
                SRSHandler.logger.log(f'SRS file did not exist when trying to retrieve AR list. HARP date: {self.HARPDate}', 'HIGH')
                print("SRS File doesn't exist; aborting...")
                return list()
        
        ARs = list()
        with open(fileToGet, 'r') as srsFile:
            all_lines = srsFile.readlines()
            indexOfSunspots, indexOfNoSunspots, indexOfRegionsToReturn = -1, -1, -1

            for i, line in enumerate(all_lines):
                if 'Regions with Sunspots' in line:
                    indexOfSunspots = i
                if 'H-alpha Plages without Spots' in line:
                    indexOfNoSunspots = i
                if 'Regions Due to Return' in line:
                    indexOfRegionsToReturn = i

            # TODO: troubleshoot if indices == -1
            if indexOfSunspots == -1 or indexOfNoSunspots == -1 or indexOfRegionsToReturn == -1:
                SRSHandler.logger.log(
                    f'One of the following lines could not be found: Regions with Sunspots (index {indexOfSunspots}); \
                        H-alpha Plages without Spots (index {indexOfNoSunspots}); \
                            Regions Due to Return (index {indexOfRegionsToReturn}. \
                                SRS file path: {fileToGet}', 'HIGH')
                return list()

            self.sunspotLine = indexOfSunspots

            begIndex = indexOfSunspots+2
            if 'None' not in all_lines[begIndex]:
                while(begIndex < indexOfNoSunspots):
                    ARs.append(SRS_AR(all_lines[begIndex], year, month, day))
                    begIndex += 1
            
            begIndex = indexOfNoSunspots+2
            if 'None' not in all_lines[begIndex]:
                while(begIndex < indexOfRegionsToReturn):
                    ARs.append(SRS_AR(all_lines[begIndex], year, month, day, hasPlage=True))
                    begIndex += 1
        
        return ARs

    def getDateTime(self) -> datetime:
        """
        Gets the datetime object representing the date of the NOAA SRS file.
        More specifically, this function retrieves the date & time at which the active region locations
        in the SRS file are valid.

        Ultimately, this is used when determining solar parameters that change as a function of time.

        Assumptions: SRS dates and times will always be in UTC time with no offset.
        If this assumption is ever false, changes will need to be made to both this function 
        and CalculateProxies_Blobs.py. 
        """

        year, month, day = (self.HARPDate.year, self.HARPDate.month, self.HARPDate.day)
        
        try:
            fileToGet = self.file
            if fileToGet is None:
                print("SRS File doesn't exist; troubleshooting...")
                raise ValueError('SRS file does not exist! Returning datetime of associated HARP.')
            
            # Get the lines that contain the 'Issued' and 'Locations Valid at' dates
            with open(fileToGet, 'r') as srsFile:
                allLines = srsFile.readlines()
                dateLine = allLines[1]
                if 'Issued' not in dateLine:
                    for line in allLines:
                        if 'Issued' in line:
                            dateLine = line
                            break
                    if 'Issued' not in dateLine:
                        SRSHandler.logger.log(f'Could not locate Issued date in SRS file. SRS file path: {fileToGet}', 'MEDIUM')
                        raise Exception
                if self.sunspotLine is not None:
                    locationsValidLine = allLines[self.sunspotLine]
            
            # Parse SRS date in UTC
            srsDateString = re.split('Issued:\s+', dateLine)[-1].strip().replace('\n','')
            splitSrsDateString = srsDateString.split()
            timeStandard = splitSrsDateString[4]
            if timeStandard != 'UTC':
                raise Exception(('Solar Region Summary (SRS) file dates/times are not in UTC. '
                    'The code should be fixed ASAP to accommodate for this change. '
                    'Changes will need to be made to both SRSHandler and CalculateProxies_Blobs.py'))

            srsHour = int(splitSrsDateString[3][:2])
            srsMinute = splitSrsDateString[3][2:]
            # Reset the hour and minute to 'noon' to avoid possibility of skipping a day when using timedelta below (with leapseconds and whatnot)
            if srsHour == 24:
                fromHourOn = ' 12' + srsMinute + ' UTC'
                srsDateString = " ".join(splitSrsDateString[:3]) + fromHourOn
            srsDate = datetime.strptime(srsDateString, '%Y %b %d %H%M %Z') # Note that %Z doesn't have any semantic meaning (python quirk)

            # In odd case that hour value is 24, increment day and reset hour
            if srsHour == 24:
                srsDate += timedelta(days=1)
                srsDate = srsDate.replace(hour=0)

            # If possible, use the 'Locations Valid at' date instead of the SRS 'Issued' date
            if self.sunspotLine is not None:
                validTime = locationsValidLine.strip().replace('\n', '').lower().split('locations valid at ')[1][:-1]  # Extract UTC time without Z

                # Reset the hour and minute to 'noon' to avoid possibility of skipping a day when using timedelta (with leapseconds and whatnot)
                srsDate = srsDate.replace(hour=12, minute=0)

                # Decrement the 'Issued' date to the 'Locations Valid at' date while keeping possible year and month differences into account 
                newDay = int(validTime.split('/')[0])
                while(srsDate.day != newDay):
                    srsDate -= timedelta(days=1)

                # Set the srsDate hour and minute to that specified in the 'Locations Valid at' time
                newHour = int(validTime.split('/')[1][:2])
                if newHour == 24:   # After the 23rd hour, SRS goes to the 24th hour, which is the 0th hour of the next day
                    srsDate += timedelta(days=1)
                    newHour = 0
                newMinute = int(validTime.split('/')[1][2:])

                srsDate = srsDate.replace(hour=newHour, minute=newMinute)
                print('New date:', srsDate)
                return srsDate
            
            # In most cases, the 'Issued' date/time is only 30 minutes off from the 'Locations Valid at' date/time, so it is acceptable to use if necessary
            return srsDate

        except Exception as e:
            SRSHandler.logger.log(repr(e)+f'. SRS file path: {fileToGet}', 'MEDIUM')
            print('!'*100)
            print('An error occurred while getting the date of the SRS file; see below for details.')
            print('\t', repr(e))
            print('Returning the date of the associated HARP.')
            print('!'*100)
            return self.HARPDate
    
    def getDateOfLastDay(self) -> datetime:
        """
        Returns a datetime object representing the date of the last day (day - 1).
        """
        
        newDate = self.HARPDate
        srsHour = newDate.hour

        # In odd case that hour value is 24, increment day and reset hour
        if srsHour == 24:
            newDate = newDate.replace(hour=12)
            newDate += timedelta(days=1)
            newDate = newDate.replace(hour=0)
        
        newDate -= timedelta(days=1) # Set date to previous day

        return newDate

    def deleteSRSFile(self):
        if self.file is not None and os.path.exists(self.file):
            os.remove(self.file)