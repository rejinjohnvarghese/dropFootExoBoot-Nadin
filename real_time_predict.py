import sys
sys.path.append(r'/Users/nadeenafifi/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Sessanta 2')

import sessanta as EMGdev
import time
import numpy as np
from scipy import signal
import serial
import csv
import threading
from sklearn.linear_model import SGDClassifier
from itertools import chain
from typing import Optional
from collections import deque
from statistics import mode
import warnings
import matplotlib.pyplot as plt
from utils import ewma_vectorized_2d
import pickle
from sklearn.preprocessing import StandardScaler
import timeit

EMG_DAMP_FACTOR = 0.33

class ExoController:
    def __init__(self, classifier_kwarg_dict=None):
        if classifier_kwarg_dict is None:
            self.classifier_kwarg_dict = {}
        self.emg_sensor = EMGSensor()
        self.imu_sensor = IMUSensor()
        self.emg_thread = None
        self.imu_thread = None
        self.phase_classifier = GaitClassifier(emg_sensor=self.emg_sensor, imu_sensor=self.imu_sensor,
                                               **classifier_kwarg_dict)

    def start_acquisitions(self):
        self.emg_thread = threading.Thread(target=self.emg_sensor.update_thread, args=(), daemon=True)
        self.emg_thread.start()
        self.imu_thread = threading.Thread(target=self.imu_sensor.update_thread, args=())
        self.imu_thread.start()

    def prediction_loop(self):

        plot_time = []
        plot_pred = []
        plot_features = []
        plot_gyro = []
        time.sleep(5)
        while True:
            try:
                self.phase_classifier.step_function()
                plot_time.append(timeit.default_timer())
                plot_pred.append(self.phase_classifier.get_majority_voting())
                plot_features.append(self.phase_classifier.cur_features)
                plot_gyro.append(self.phase_classifier.smoothed_imu[0])

                if self.phase_classifier.imu_sensor.runningTime > 30:
                    with open(r"emg_1000_stack3.pickle", "wb") as output_file:
                        e = pickle.dump([plot_time, plot_features, plot_pred, plot_gyro, self.emg_sensor.stored_data] ,output_file)
                    print("All done!")
                    break

            except BaseException as e:
                print("disconnecting")
                self.emg_sensor.should_disconnect = True
                time.sleep(0.5)
                raise e

    def emg_loop(self):
        while True:
            print(self.emg_sensor.cur_emg.shape)

    def update_model(self):
        raise NotImplementedError()

class IMUSensor:
    def __init__(self, mpu_arduino_address='COM4',baudrate=38400, memory_size=10000):
        self.mpu_arduino = serial.Serial(mpu_arduino_address, baudrate)
        self.Gyro_X = 0
        self.Gyro_Y = 0
        self.Gyro_Z = 0
        self.Acc_X = 0
        self.Acc_Y = 0
        self.Acc_Z = 0
        self.connect_arduino()
        self.imu_del_t = 0
        self.runningTime = 0
        self.stored_data = np.empty((memory_size, 6))
        self.stored_data[:] = np.nan
        self.memory_idx = 0
        self.totalTime = []

    def connect_arduino(self):
        while True:
            while (self.mpu_arduino.inWaiting == 0):
                pass
            arduinostring = self.mpu_arduino.readline()
            print(arduinostring)
            if arduinostring == b"All clear!\r\n":
                break

    def update_imu_data(self):
        while (self.mpu_arduino.inWaiting == 0):
            pass
        self.mpu_arduino.write(str.encode('1'))
        arduinostring = self.mpu_arduino.readline()
        arduinostring = str(arduinostring, encoding="utf-8")
        dataArray = arduinostring.split(',')
        self.Gyro_X = float(dataArray[0])
        self.Gyro_Y = float(dataArray[1])
        self.Gyro_Z = float(dataArray[2])
        self.Acc_X = float(dataArray[3])
        self.Acc_Y = float(dataArray[4])
        self.Acc_Z = float(dataArray[5])
        if self.memory_idx < self.stored_data.shape[0]:
            self.stored_data[self.memory_idx, :] = dataArray
            self.memory_idx += 1
        return [self.Gyro_X,self.Gyro_Y,self.Gyro_Z,self.Acc_X,self.Acc_Y,self.Acc_Z]

    @property
    def cur_imu_data(self):
        return [self.Gyro_X,self.Gyro_Y,self.Gyro_Z,self.Acc_X,self.Acc_Y,self.Acc_Z]

    def is_memory_full(self):
        return self.memory_idx <= self.stored_data.shape[0]

    def update_thread(self):
        t_0 = timeit.default_timer()

        while True:
            self.update_imu_data()
            self.imu_del_t = timeit.default_timer() - t_0
            self.runningTime += self.imu_del_t
            t_0 = timeit.default_timer()
            self.totalTime.append(t_0)

    def save_file(self,filename='IMU.csv'):
        rows = self.stored_data
        with open(filename, 'a+') as csvfile:
            fieldnames = ("Gyro_X", "Gyro_Y", "Gyro_Z", "Acc_X", "Acc_Y", "Acc_Z")
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for row in rows:
                writer.writerow(row)

class EMGSensor:
    def __init__(self,window_size=100,memory_size=400,  **kwargs):
        self.emg = EMGdev.Sessantaquattro(buffsize=window_size, **kwargs)
        self.emg.connect_to_sq()
        self.stored_data = np.empty((memory_size*window_size, 64))
        self.memory_idx = 0
        self.window_size = window_size
        self.cur_emg = np.zeros((200, 64))
        self.emg_del_t = 0
        self.runningTime = 0
        self.fs_sq = self.emg._sample_frequency
        self.should_disconnect = False
        self.totalTime = []

    def update_emg(self):
        emg_data, bin_data = self.emg.read_emg()
        self.cur_emg = emg_data
        if self.memory_idx*self.window_size < self.stored_data.shape[0]:
            self.stored_data[self.memory_idx*self.window_size:(self.memory_idx+1)*self.window_size, :] = emg_data[:,:64]

            self.memory_idx += 1
        return self.cur_emg[:, :64]

    def get_envelope(self, low_cutoff=20, high_cutoff = 500, lp_cutoff = 3):
        ##------- BANDPASS FILTER -------##
        fs = self.fs_sq
        b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='bandpass', fs=fs)
        emg_bandpass = signal.filtfilt(b, a, self.cur_emg)

        ##------- FULL WAVE RECTIFICATION -------##
        emg_rectified = abs(emg_bandpass)

        ##------- LOWPASS FILTER -------##
        # create lowpass filter and apply to rectified signal to get EMG envelope
        b2, a2 = signal.butter(2, lp_cutoff / (fs / 2), btype='lowpass')
        emg_envelope = signal.filtfilt(b2, a2, emg_rectified)
        return emg_envelope

    def get_rms(self):
        emg_zeromean = self.cur_emg - np.mean(self.cur_emg, axis=0)[None, :]
        emg_rms = np.sqrt(np.mean(emg_zeromean * emg_zeromean))
        return emg_rms

    def update_thread(self):
        t_0 = timeit.default_timer()
        while True:
            self.update_emg()
            self.emg_del_t = timeit.default_timer() - t_0
            self.runningTime += self.emg_del_t
            t_0 = timeit.default_timer()
            self.totalTime.append(t_0)
            if self.should_disconnect:
                print("emg sensor disconnecting!")
                self.emg.disconnect_from_sq()

    @staticmethod
    def to_feature_vector(emg_window, range_h):
        emg_hist = np.histogram(np.clip(emg_window[:], *range_h), bins=9)[0]
        zm = emg_window - np.mean(emg_window, axis=0)[None, :]
        rms = np.sqrt(np.mean(zm*zm))
        bp = zm[:, 31] - zm[:, 33]
        bp_rms = np.sqrt(np.mean(bp*bp))

        # [0, 1, 2, ..., 199] -> RMS || [0, 1, 2, ... 99] -> RMS1 ; [100, 101, ... 199] -> RMS2 => [RMS1 RMS2]
        emg_stack1 = zm[:50, :]
        emg_stack2 = zm[50:100, :]
        rms_stack1 = np.sqrt(np.mean(emg_stack1*emg_stack1))
        rms_stack2 = np.sqrt(np.mean(emg_stack2*emg_stack2))
        rms_diff = rms_stack2 - rms_stack1
        bp_stack1 = zm[:50, 31] - zm[:50, 33]
        bp_stack2 = zm[50:100, 31] - zm[50:100, 33]
        rms_stack3 = np.sqrt(np.mean(bp_stack1*bp_stack1))
        rms_stack4 = np.sqrt(np.mean(bp_stack2*bp_stack2))
        bprms_diff = rms_stack4 - rms_stack3

        return [rms, bp_rms, rms_diff, bprms_diff]

    @property
    def range_h(self):
        return (self.stored_data[:].min()*0.8, self.stored_data[:].max()*0.8)

    def __del__(self):
        if not self.should_disconnect:
            self.emg.disconnect_from_sq()

class GaitClassifier:
    model: Optional[SGDClassifier]
    imu_sensor: IMUSensor
    emg_sensor: EMGSensor

    def __init__(self, imu_sensor=None, emg_sensor=None, n_iter=500, phase_memory=9, alpha=1/11, debug=False,
                 feature_mask=None, should_smooth_emg=False, peak_distance=90):
        if emg_sensor is None:
            self.emg_sensor = EMGSensor()
        else:
            self.emg_sensor = emg_sensor
        if imu_sensor is None:
            self.imu_sensor = IMUSensor()
        else:
            self.imu_sensor = imu_sensor
        self.n_iter = n_iter
        self.counter = 0
        self.model = None
        self.cur_phase = 0
        self.emg_thread = None
        self.phase_deque = deque(maxlen=phase_memory)
        self.arr_features = []
        self.alpha = alpha
        self.smoothed_imu = None
        self.debug = debug
        self.feature_mask = slice(0, None) if feature_mask is None else feature_mask
        self.cur_features = np.zeros(self.feature_mask.stop-self.feature_mask.start)
        self.normalizer = StandardScaler()
        self.should_smooth_emg = should_smooth_emg
        self.smoothed_emg = None
        self.peak_distance =peak_distance

    @staticmethod
    def process_observations(arr, distance=90):
        arr = np.array(arr)
        imu_arr = arr[:, :6]
        #imu_arr = ewma_vectorized_2d(imu_arr, alpha=alpha, axis=0)
        #if smooth_emg:
         #   arr[:, 6:] = ewma_vectorized_2d(arr[:, 6:], alpha=alpha/EMG_DAMP_FACTOR, axis=0)
        def crossings_nonzero_neg2pos(data):
            neg = data < 0
            HS = (neg[:-1] & ~neg[1:]).nonzero()[0]
            return HS

        # identify heel strikes and toe off points from IMU data
        HS_idx =  crossings_nonzero_neg2pos(imu_arr[:,0])
        TO_idx = signal.find_peaks(imu_arr[:,0], height=2000, distance=distance)[0]
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
        features = np.hstack([imu_arr, arr[:, 6:]])

        crop = slice(min(chain(TO_idx, HS_idx)), max(chain(HS_idx, TO_idx)))
        features = features[crop, :]
        assert labels.shape[0] == features.shape[0]

        return features, labels # change here

    def update_sensors(self):
        warnings.warn("Sensors should be updating themselves through their threads!")
        cur_emg = self.emg_sensor.update_emg()
        cur_imu = self.imu_sensor.update_imu_data()
        return cur_emg, cur_imu

    @staticmethod
    def fit_model(clf, features,labels,th=15):
        if clf is None:
            clf = SGDClassifier(loss="hinge")
        model = clf.partial_fit(features[:-th, :], labels[th:], classes=labels)  # th is time horizon meaning how much in the future the model is predicting.
        return model

    def classifying_loop(self):
        while True:
            self.step_function()

    def step_function(self, delta_time=1):
        if self.smoothed_imu is None:
            self.smoothed_imu = self.imu_sensor.cur_imu_data


        delta_time = np.clip(delta_time, 20*self.alpha, 0.1)

        cur_emg, cur_imu = self.emg_sensor.cur_emg, self.imu_sensor.cur_imu_data

        emg_features = self.emg_sensor.to_feature_vector(cur_emg[:, :64], self.emg_sensor.range_h)

        if self.should_smooth_emg:
            if self.smoothed_emg is None:
                self.smoothed_emg = emg_features
            self.smoothed_emg = (1 - self.alpha/delta_time/EMG_DAMP_FACTOR) * np.array(self.smoothed_emg) + self.alpha/delta_time/EMG_DAMP_FACTOR * np.array(emg_features)


        self.smoothed_imu = (1-self.alpha/delta_time)*np.array(self.smoothed_imu) + self.alpha/delta_time*np.array(cur_imu)
        pred_arr = np.concatenate([self.smoothed_imu, emg_features if not self.should_smooth_emg else self.smoothed_emg])[None, :]

        arr = np.concatenate([self.smoothed_imu, emg_features if not self.should_smooth_emg else self.smoothed_emg])
        self.arr_features.append(arr)

        self.normalizer.partial_fit(pred_arr)
        pred_arr = self.normalizer.transform(pred_arr)
        pred_phase = GaitClassifier.get_classification(pred_arr[:, self.feature_mask], self.model) # change here

        self.cur_features = pred_arr[:, self.feature_mask]

        self.counter += 1
        # print(f"Counter is {self.counter}, n_iter is {self.n_iter}")
        if self.counter % self.n_iter == 0:  # fit model and reset memory every n iterations
            print("Resetting counter")
            self.counter = 0
            if self.debug is True:
                return pred_phase
            try:
                features, labels = GaitClassifier.process_observations(self.arr_features, distance=self.peak_distance/delta_time)
            except BaseException as e:
                features, labels = GaitClassifier.process_observations(self.arr_features, distance=self.peak_distance/delta_time)
                raise e
            self.normalizer.partial_fit(features)
            features = self.normalizer.transform(features)
            self.model = GaitClassifier.fit_model(self.model, features[:, self.feature_mask], labels)


        self.cur_phase = pred_phase
        self.phase_deque.append(pred_phase)
        return pred_phase

    def get_majority_voting(self):
        phase_mode = mode(self.phase_deque)
        return phase_mode

    @staticmethod
    def get_classification(arr, model):
        if model is None:
            return -1
        gait_phase = model.predict(arr)[0]
        return gait_phase

def main():
    controller = ExoController(classifier_kwarg_dict={"n_iter": 1000, "feature_mask": slice(6, 10), "phase_memory": 15,
                                                      "should_smooth_emg": True, "alpha": 1, "peak_distance": 0.5 })
    controller.start_acquisitions()
    controller.prediction_loop()
    pass


if __name__ == '__main__':
    main()

