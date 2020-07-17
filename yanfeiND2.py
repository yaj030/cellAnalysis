from nd2reader import ND2Reader
from nd2reader.parser import Parser
class yanfeiND2reader(ND2Reader):
    """ wrapper for the ND2 reader.
    The reason for a new wrapper is that ND2reader does not work well with truncated ND2 file, positions 
    were sometimes truncated
    This is the main class: use this to process your .nd2 files.
    """

    def __init__(self, filename):
        super(ND2Reader, self).__init__()
        self.filename = filename

        # first use the parser to parse the file
        self._fh = open(filename, "rb")
        self._parser = Parser(self._fh)

        # Setup metadata
        self.metadata = self._parser.metadata

        # Set data type
        self._dtype = self._parser.get_dtype_from_metadata()

        #
        self.metadata['fields_of_view'] = range(0,4)

        # Setup the axes
        self._setup_axes()

        # Other properties
        self._timesteps = None