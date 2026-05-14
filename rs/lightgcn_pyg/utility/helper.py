import os
import re

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    import numpy as np
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if best_value is None or len(best_value) == 0:  # If best_value is empty, initialize it
        best_value = np.array(log_value)
        stopping_step = 0
        should_stop = False
    else:
        if expected_order == 'acc':
            # Check if all elements in log_value are greater than or equal to their counterparts in best_value
            if all(lv >= bv for lv, bv in zip(log_value, best_value)):
                stopping_step = 0
                best_value = np.array(log_value)
            else:
                stopping_step += 1
        elif expected_order == 'dec':
            # Check if all elements in log_value are less than or equal to their counterparts in best_value
            if all(lv <= bv for lv, bv in zip(log_value, best_value)):
                stopping_step = 0
                best_value = np.array(log_value)
            else:
                stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is triggered at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop
