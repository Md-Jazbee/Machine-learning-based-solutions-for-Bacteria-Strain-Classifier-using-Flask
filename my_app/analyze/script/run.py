import os
def predict(b,a):
    if b=="sensitive":
        s="python3 PREDICTION_CODE/command_line.py predict --sensitive " + str(a)
    elif b=="rapid":
        s="python3 PREDICTION_CODE/command_line.py predict --rapid " + str(a)

    os.system(s)