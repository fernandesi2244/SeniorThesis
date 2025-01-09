"""
Used to represent an active region (AR) entry within a NOAA Solar Region Summary (SRS)
file.

This makes accessing information about each AR more convenient in CalculateProxies. 
Additionally, by recording all the attributes of the AR, the program makes it so that,
if the programmer decides another attribute of an AR should be used in associating ARs
with blobs, then the change can easily be implemented without worrying about changes 
in the parsing of the SRS file. This ties into the final benefit, which is the fact
that changes in the format of the AR entries within an SRS file by NOAA can be addressed
in one place without affecting other pieces of code.

Developers
----------
Ian Fernandes: fernandesi2244@gmail.com

NASA JSC, Space Medicine Operations Division (SD2), Space Radiation Analysis Group (SRAG)

Last Updated 2020
"""

class SRS_AR(object):
    """
    Represents an active region from the list of active regions in a NOAA Solar Region Summary file.
    """

    def __init__(self, SRSLine, year, month, day, hasPlage=False):
        """
        Constructs an SRS Active Region object
        """

        self.arNum = None
        self.location = None
        self.carrLongitude = None
        self.area = None
        self.zurichClassification = None
        self.longitudinalExtent = None
        self.numSunspots = None
        self.magType = None

        # A plage has no sunspots; we must deal with these ARs differently during identification/training
        self.hasPlage = hasPlage
        self.distanceFromBlob = None

        # Year, month, and day of SRS file
        self.year = year
        self.month = month
        self.day = day
        
        self.parseLine(SRSLine)

    def parseLine(self, SRSLine):
        """
        Parses a line of the SRS file and stores relevant information about the active region.
        """

        line = SRSLine.split()

        self.arNum = int(line[0])
        # Only add 10,000 to AR # in the following 3 cases (in around June of 2002, AR # passed 9999 and cycled back to 0000)
        addTenThousand = self.year > 2002 or (self.year == 2002 and self.month >=7) or (self.year == 2002 and self.month == 6 and 10_000-self.arNum >= 100)
        self.arNum += 10_000 if addTenThousand else 0

        self.location = line[1]
        self.carrLongitude = int(line[2])

        if not self.hasPlage:
            self.area = int(line[3])
            self.zurichClassification = line[4]
            self.longitudinalExtent = int(line[5])
            self.numSunspots = int(line[6])
            # Sometimes, there is no mag type in the SRS file
            if len(line) > 7:
                self.magType = line[7]
    
    def setDistanceFromBlob(self, distance):
        """
        Sets distance of AR coordinates from blob for AR identification purposes.

        If the AR is within the blob, this is the distance from the blob's centroid.
        If the AR is outside the blob, this is the approximate closest distance to the
        blob's boundary.
        """
        
        self.distanceFromBlob = distance

    def __repr__(self):
        return str(self.arNum)