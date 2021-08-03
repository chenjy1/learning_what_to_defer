import sys
import time
class Logger():
    def __init__(self, filename = None, enable_terminal = 1, enable_log = 1):

        if filename is None:
            OWO = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        else:
            OWO = filename + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) 

        OWO = "log/" + OWO + ".txt"
        self.terminal = sys.stdout
        self.log = open(OWO, "w+")
        self.enable_terminal = enable_terminal
        self.enable_log = 1

    def write(self, message):
        if self.enable_terminal:
            self.terminal.write(message)
        if self.enable_log:
            self.log.write(message)

    def flush(self):
        pass
