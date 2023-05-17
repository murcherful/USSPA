import time


def get_current_time():
    return str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def second2time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

class ProgressBar():
    def __init__(self, length):
        self.length = length
        self.As = '#'
        self.Bs = '-'
        self.A = ''
        self.B = ''
        self.cS = 0
        self.S = '-\\|/'
        for i in range(self.length):
            self.A += self.As
            self.B += self.Bs
    
    def get_bar(self, p):
        Ap = int(self.length*p)
        Bp = self.length - Ap
        if Bp > 0:
            text = '<' + self.A[:Ap] + self.S[self.cS] + self.B[:Bp] + '>'
        else:
            text = '<' + self.A[:Ap] + '>' 
        self.cS = (self.cS + 1) % 4 
        return text

        
