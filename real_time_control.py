import time
from ctypes import *
from real_time_predict import ExoController
import threading
import numpy as np
import warnings
import pickle
from epos_control import EPOS4USB_sensorData
import timeit


class MannequinController(ExoController):
    def __init__(self, **kwargs):
        super(MannequinController, self).__init__(**kwargs)
        self.motor_sensor = EPOS4USB_sensorData(b'USB0',0,1000000,500)
        self.starting_pos = self.motor_sensor.getMotorPosition()

    def start_acquisitions(self):
        self.emg_thread = threading.Thread(target=self.emg_sensor.update_thread, args=(), daemon=True)
        self.emg_thread.start()
        self.imu_thread = threading.Thread(target=self.imu_sensor.update_thread, args=(), daemon=True)
        self.imu_thread.start()
        self.motor_sensor.daemon = True
        self.motor_sensor.speed = 1500
        self.motor_sensor.start()


    def control_loop(self, timeout=30):

        plot_time = []
        plot_pred = []
        plot_gyro = []
        plot_motorpos = []
        plot_string_pot = []
        plot_features =[]
        plot_load = []
        time.sleep(3)
        delta_time = 0.01
        t_0 = timeit.default_timer()
        while True:
            self.phase_classifier.step_function(delta_time=delta_time)
            
            predicted = self.phase_classifier.get_majority_voting()
            plot_time.append(timeit.default_timer())
            plot_pred.append(predicted)
            plot_gyro.append(self.phase_classifier.smoothed_imu[0])
            plot_motorpos.append(self.motor_sensor.motorPosition)
            plot_string_pot.append(self.motor_sensor.anInV_1)
            plot_features.append(self.phase_classifier.cur_features)
            plot_load.append(self.motor_sensor.anInV_0)
            
            if predicted == 1:
                self.set_target_motor_position(-65000)
            else:
                self.set_target_motor_position(0)

            if self.phase_classifier.imu_sensor.runningTime > timeout:
                with open(r"motor_data_both_thursday_WR03.pickle", "wb") as output_file:
                    data = {"time": plot_time, "motor_time": self.motor_sensor.totalTime,"imu time": self.imu_sensor.totalTime, "emg time":self.emg_sensor.totalTime, "prediction": plot_pred, "gyro x":  plot_gyro, "motor pos": plot_motorpos, "string pot": plot_string_pot, "features": plot_features, "load":  plot_load, "emg_memory": self.emg_sensor.stored_data}
                    pickle.dump(data, output_file)
                break
         
            delta_time = timeit.default_timer() - t_0
            t_0 = timeit.default_timer()
        print("All done!")

    def set_target_motor_position(self, absolute_pos):

        if absolute_pos < -65000:
            warnings.warn("Too large dorsiflexion!")
            return
        if absolute_pos > 0:
            warnings.warn("Plantarflexion not supported.")
            return
        # if not self.motor_sensor.getMovementState():
        #     print("Target not reached yet!")
        #     return
        self.motor_sensor.target_position = absolute_pos

def main():
    testing_controller = MannequinController(classifier_kwarg_dict={"n_iter": 1000, "feature_mask": slice(0, 10), "phase_memory": 30,
                                                      "should_smooth_emg": True, "alpha": 1/2500, "peak_distance":0.75})
    testing_controller.start_acquisitions()
    testing_controller.control_loop()

if __name__ == '__main__':
    main()