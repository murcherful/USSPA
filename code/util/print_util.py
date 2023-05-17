from termcolor import cprint
import sys
import os


LABEL_LINE = '**='
for i in range(5):
    LABEL_LINE += LABEL_LINE
LABEL_LINE += '**'

def clear_print(x):
    print('')
    print(x)
    print('')

def print_info(text, prefix=''):        # add '\n' automatically
    text = str(text)
    text = '<INFO> ' + prefix + ' ' + text 
    #cprint(text, 'green', 'on_white')
    cprint(text, 'green')
    return text

def print_output(text, prefix=''):
    text = str(text)
    text = '<#OUT> ' + prefix + ' ' + text
    #cprint(text, 'blue', 'on_white')
    cprint(text, 'blue')
    return text

def print_warn(text, prefix=''):
    text = str(text)
    text = '<WARN> ' + prefix + ' ' + text
    #cprint(text, 'yellow', 'on_white')
    cprint(text, 'yellow')
    return text

def clear_print_line(text, color='blue'):
    clear_line()
    cprint(text, color)


def move_up(line_num):
    if line_num > 0:
        print('\033[%dA' % line_num, end='', flush=True)

def move_down(line_num):
    if line_num > 0:
        print('\033[%dB' % line_num, end='', flush=True)

def clear_line():
    print('\033[K', end='', flush=True)

def save_cur():
    print('\033[s', end='', flush=True)
    
def restore_cur():
    print('\033[u', end='', flush=True)
    
def flash_label():
    save_cur()
    print('\033[5m'+LABEL_LINE+'\033[0m', end='', flush=True)
    restore_cur()


class PrintLogger():
    def __init__(self, log_dir, is_restore, file_name='term_out.log', prefix=''):
        self.log_dir = log_dir
        self.prefix = prefix
        if is_restore:
            self.out_file = open(os.path.join(self.log_dir, file_name), 'a')
        else:
            self.out_file = open(os.path.join(self.log_dir, file_name), 'a')
        self.MAX_line = None
        self.current_line = 0
        self.append_num = 0


    def __del__(self):
        self.out_file.close()

    
    def set_max_line(self, num):
        self.MAX_line = num

    def normal(self):
        move_down(self.MAX_line-self.current_line+self.append_num)
        self.MAX_line = None


    def check_position(self):
        if self.MAX_line != None: 
            if self.current_line >= self.MAX_line:
                self.current_line = 1
                self._clear_line()
                move_up(self.MAX_line) 
            else:
                self.current_line += 1
   
    def _clear_line(self):
        if self.MAX_line != None:
            clear_line()  

    def _flash_label(self):
        if self.MAX_line != None:
            flash_label()

    def _get_prefix(self):
        if self.MAX_line != None:
            return self.prefix + ' (%02d)' % self.current_line
        else:
            return self.prefix

    def log_info(self, text):        # add '\n' automatically
        self.check_position()
        text = str(text)
        self._clear_line()
        text = print_info(text, self._get_prefix())
        self._clear_line()
        self._flash_label()
        self.out_file.write(text+'\n')
        self.out_file.flush()

    def log_output(self, text):
        self.check_position()
        text = str(text)
        self._clear_line()
        text = print_output(text, self._get_prefix())
        self._clear_line()
        self._flash_label()
        self.out_file.write(text+'\n')
        self.out_file.flush()

    def log_warn(self, text):
        text = str(text)
        text = print_warn(text, self.prefix)
        self.out_file.write(text+'\n')
        self.out_file.flush()

    def log_file(self, text):
        if isinstance(text, list):
            for t in text:
                self.out_file.write(t+'\n')
        else:
            self.out_file.write(text+'\n')
        self.out_file.flush()

    def append_print(self, text, index, to_file=False):
        if self.MAX_line != None:
            self.append_num = max(self.append_num, index)
            save_cur()
            s = self.MAX_line-self.current_line+index-1
            move_down(s)
            self._clear_line()
            print(text, end='')
            restore_cur()
        else:
            print(text)
        if to_file:
            self.out_file.write(text+'\n')
            self.out_file.flush()
        

