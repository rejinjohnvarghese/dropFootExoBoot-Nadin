import pickle
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from itertools import chain

def main():

    # sns.set_theme()
    # sns.set_context('poster')

    with open(r"C:\Users\rejin\Desktop\EXO_files\motor_data_both_thursday_WR02.pickle", "rb") as input_file:
        e: Dict = pickle.load(input_file)

    e = {k: np.array(e) for k, e in e.items()}

    HS_idx =  crossings_nonzero_neg2pos(e['gyro x'])
    TO_idx = signal.find_peaks(e['gyro x'], height=2000, distance=200)[0]
    label = []
    heel_first = min(HS_idx) < min(TO_idx)
    heel_slice = slice((1 if heel_first else 0), None)
    toe_slice = slice((0 if heel_first else 1), None)

    for heel_strike, toe_off in zip(np.repeat(HS_idx, 2)[heel_slice], np.repeat(TO_idx, 2)[toe_slice]):
        if (toe_off < heel_strike):
            label.append(np.ones(heel_strike - toe_off))
        else:
            label.append(np.zeros(toe_off - heel_strike))

    labels = np.concatenate(label) # 0 stance, 1 swing


    crop = slice(min(chain(TO_idx,HS_idx)), max(chain(TO_idx,HS_idx)))
    predicted = e['prediction']
    predicted = [np.nan if x == -1 else x for x in predicted]
    predicted = predicted[crop]

    acc = np.sum(predicted[2930:] == np.array(label)[2930:])/len(label[2930:])
    print("ACCURACY IS:",acc*100)

    plt.plot(e['gyro x'])
    plt.plot(e['prediction']*2200)

    plt.figure()
    plt.plot(e['motor pos']/300)
    plt.plot(e['string pot']-6220)

    plt.figure()
    plt.plot(e['gyro x']/2200)
    plt.plot(e['features'][:, 0, 7])


    plt.tight_layout()
    plt.show()


def crossings_nonzero_neg2pos(data):
    neg = data < 0
    HS = (neg[:-1] & ~neg[1:]).nonzero()[0]
    return HS
if __name__ == '__main__':
    main()
