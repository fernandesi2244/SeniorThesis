"""
Logs any worrisome or unexpected events to a log file so that problems can
be more efficiently troubleshooted later.

Developers
----------
Ian Fernandes: fernandesi2244@gmail.com

NASA JSC, Space Medicine Operations Division (SD2), Space Radiation Analysis Group (SRAG)

Last Updated 2021
"""

import os
import sys
import pathlib
import datetime

rootDir = pathlib.Path(__file__).resolve().parent.absolute()

class Logger(object):
    """
    An object used to log important information while the program is operating so that troubleshooting
    can be more efficient.

    See bottom of file for example usage.
    """

    def __init__(self, fileName='DataminingLog.txt', inOperationalMode: bool=False):
        """
        Creates a new Logger object with the specified log file name.
        This name can be changed for when the Logger object is used in 
        different settings so that different categories of logs can be 
        kept separate if desired.

        :param fileName: the file name of the log file to write to (defaults to DataminingLog.txt)
        """

        self.file = os.path.join(rootDir, 'OutputData', 'Logs', fileName)

        if not os.path.exists(self.file):
            if not os.path.exists(os.path.join(rootDir, 'OutputData', 'Logs')):
                os.makedirs(os.path.join(rootDir, 'OutputData', 'Logs'))
            self.log(f'Log file created at {datetime.datetime.now()}')
    
    def log(self, message: str, priority: str = 'LOW'):
        """
        Logs the specified message with its priority to the log file
        so that critical issues can be located easily.

        Recommended priority codes:
        HIGH, MEDIUM, LOW

        LOW: Interesting scenarios that might want to be addressed in the future
        MEDIUM: Slightly concerning problems that should be looked at when possible
        HIGH: Highly concerning problems that should be checked and addressed ASAP 

        :param message: the message to log
        :param priority: the priority of the message (defaults to LOW)
        """
        try:
            with open(self.file, 'a') as logFile:
                if priority.lower() == 'high':
                    logFile.write('!'*100+'\n')

                logFile.write(f'Priority: {priority.upper()} | Message: {message}\n')

                if priority.lower() == 'high':
                    logFile.write('!'*100+'\n')
        except Exception as e:
            with open(os.path.join(rootDir, 'OutputData', 'Logs', 'LoggerProblems.txt'), 'a') as logProbFile:
                logProbFile.write(f'Logger failed with the following exception: {repr(e)}\n')
    
    def clearLog(self):
        """
        Clears the log file associated with this Logger object.
        """

        with open(self.file, 'r+') as logFile:
            logFile.truncate(0)

'''
EXAMPLE USAGE:
--------------
logger = Logger()
logger.log("No neutral line segments!")
logger.log('Skipped a file', 'MEDIUM')
logger.log('SRS file could not be retrieved for this HARP date!', 'HIGH')
logger.clearLog()
'''