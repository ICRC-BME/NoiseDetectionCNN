import time
class timer(object):
    def __init__(self, name=None,verbose=True):
        self.name = name
        self.verbose=verbose

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            if self.name:
                print("["+self.name+"] Elapsed: "+str(time.time() - self.tstart)+" seconds.")
            else:
                print("Elapsed: " + str(time.time() - self.tstart)+" seconds.")