import pickle
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from itertools import chain

def main():

    # sns.set_theme()
    # sns.set_context('poster')

    with open(r"C:\Users\rejin\Desktop\EXO_files\motor_data_emg_thursday_WR03.pickle", "rb") as input_file:
        e: Dict = pickle.load(input_file)

    e = {k: np.array(e) for k, e in e.items()}


    plt.plot(e['gyro x'])
    plt.plot(e['prediction']*2200)

    plt.figure()
    plt.plot(e['motor pos']/300)
    plt.plot(e['string pot']-6220)

    plt.figure()
    plt.plot(e['gyro x']/2200)
    plt.plot(e['features'][:, 0, 1])


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
