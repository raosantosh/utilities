class CircularFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def readline(self):
        if (self.file is None):
            self.file = open(self.filename, "r")

        p_line = self.file.readline()

        if p_line == "":
            self.file.close()
            self.file = open(self.filename, "r")
            p_line = self.file.readline()
        return p_line

    def close(self):
        self.file.close()
        self.file = None