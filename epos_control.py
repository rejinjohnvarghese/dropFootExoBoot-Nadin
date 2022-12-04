from array import array
from select import select
from threading import Lock, Thread
import matplotlib.pyplot as plt
import time
from ctypes import *
import multiprocessing

# EPOS Command Library path
path='.\EposCmd64.dll'

# Load library
cdll.LoadLibrary(path)
epos = CDLL(path)
print("EPOS Library loaded")

threadLock = Lock()

# class to get sensing data from EPOS4 motor controller
class EPOS4USB_sensorData(Thread):
    # constructor (initialize variables)
    def __init__(self,USBIdx,nodeID,baudRate,timeout):
        super().__init__()
        self.nodeID = nodeID
        self.pErrorCode = c_uint()
        self.motorPosition = 0
        self.motorVelocity = 0
        self.motorCurrent = 0
        self.anInV_0 = 0
        self.anInV_1 = 0
        self.digIn = [1 for i in range(10)]
        self.sensorData_del_t = 1
        self.runningTime = 0

        self.timeout = 1000
        self.motor_ready_flag = False

        self.target_position = 0
        self.speed = 500

        self.totalTime = []
        # self._motorPosSteps = 0
        # self._motorPosSteps_old = 0
        # self._motorVelSteps = 0
        # self._motorVelSteps_old = 0
        # self._motorCurrSteps = 0
        # self._motorCurrSteps_old = 0
        # self._anIn_0_Steps = 0
        # self._anIn_0_Steps_old = 0
        # self._anIn_1_Steps = 0
        # self._anIn_1_Steps_old = 0
        # self._digIn_Steps = 0
        # self._digIn_Steps_old = 0
 
        # Initiating connection and setting motion profile
        self.keyHandle = epos.VCS_OpenDevice(b'EPOS4', b'MAXON SERIAL V2', b'USB', USBIdx, byref(self.pErrorCode)) # specify EPOS version and interface
        print("Device opened")  
        epos.VCS_SetProtocolStackSettings(self.keyHandle, baudRate, timeout, byref(self.pErrorCode)) # set baudrate
        epos.VCS_ClearFault(self.keyHandle, self.nodeID, byref(self.pErrorCode)) # clear all faults

    def setEnableState(self):
            epos.VCS_SetEnableState(self.keyHandle, self.nodeID, byref(self.pErrorCode)) # enable device


    # Query motor position
    def getMotorPosition(self):
        # self._motorPosSteps = self._motorPosSteps_old + 1
        pPositionIs=c_long()
        ret=epos.VCS_GetPositionIs(self.keyHandle, self.nodeID, byref(pPositionIs), byref(self.pErrorCode))
        self.motorPosition = pPositionIs.value
        return self.motorPosition # motor position

    def getMovementState(self):
        # self._motorPosSteps = self._motorPosSteps_old + 1
        pMovementState = c_bool()
        ret=epos.VCS_GetMovementState(self.keyHandle, self.nodeID, byref(pMovementState), byref(self.pErrorCode))
        return pMovementState.value

    def moveToPosition(self, value):
        # raise NotImplementedError()
        target_pos = value
        epos.VCS_MoveToPosition(self.keyHandle, self.nodeID, target_pos, c_bool(True), c_bool(True), byref(self.pErrorCode))
        #VCS_WaitForTargetReached(self.KeyHandle, self.NodeId, self.timeout, byref(self.pErrorCode))

        #epos.VCS_set(value)

    def activateProfilePositionMode(self):
        epos.VCS_ActivateProfilePositionMode(self.keyHandle, self.nodeID, byref(self.pErrorCode))

    def setPositionProfile(self, vel=1000//2, acc=10000, dec=10000):

        epos.VCS_SetPositionProfile(self.keyHandle, self.nodeID, vel, acc, dec, byref(self.pErrorCode))

    def ready_motor(self, **profile_kwargs):
        if self.motor_ready_flag:
            return
        self.setEnableState()
        self.activateProfilePositionMode()
        self.setPositionProfile(vel=self.speed)
        self.motor_ready_flag = True
        

    def MoveToPositionSpeed(self, target_position,target_speed, acceleration=10000, deceleration=10000):
        # while True:
        self.ready_motor()
        if target_speed != 0:
            epos.VCS_MoveToPosition(self.keyHandle, self.nodeID, target_position, True, True, byref(self.pErrorCode)) # move to position
        elif target_speed == 0:
            epos.VCS_HaltPositionMovement(self.keyHandle, self.nodeID, byref(self.pErrorCode)) # halt motor
            # true_position = self.getMotorPosition()
            # if abs(true_position - target_position) < 500:
            #     break

        

    # # Query motor position thread
    # def getMotorPositionThread(self):
    #     while True:
    #         # self._motorPosSteps = self._motorPosSteps_old + 1
    #         pPositionIs=c_long()
    #         ret=epos.VCS_GetPositionIs(self.keyHandle, self.nodeID, byref(pPositionIs), byref(self.pErrorCode))
    #         self.motorPosition = pPositionIs.value
    #         return self.motorPosition # motor position


    # Query motor velocity
    def getMotorVelocity(self):
        pVelocityIs=c_long()
        ret=epos.VCS_GetVelocityIs(self.keyHandle, self.nodeID, byref(pVelocityIs), byref(self.pErrorCode))
        # ret=epos.VCS_GetVelocityIsAveraged(self.keyHandle, self.nodeID, byref(pVelocityIs), byref(self.pErrorCode))
        self.motorVelocity = pVelocityIs.value
        return self.motorVelocity # motor velocity

    # # Query motor velocity thread
    # def getMotorVelocityThread(self):
    #     while True:
    #         pVelocityIs=c_long()
    #         ret=epos.VCS_GetVelocityIs(self.keyHandle, self.nodeID, byref(pVelocityIs), byref(self.pErrorCode))
    #         # ret=epos.VCS_GetVelocityIsAveraged(self.keyHandle, self.nodeID, byref(pVelocityIs), byref(self.pErrorCode))
    #         self.motorVelocity = pVelocityIs.value
    #         return self.motorVelocity # motor velocity



    # Query motor current
    def getMotorCurrent(self):
        pCurrentIs=c_long()
        ret=epos.VCS_GetCurrentIs(self.keyHandle, self.nodeID, byref(pCurrentIs), byref(self.pErrorCode))
        # ret=epos.VCS_GetCurrentIsAveraged(self.keyHandle, self.nodeID, byref(pCurrentIs), byref(self.pErrorCode))
        self.motorCurrent = pCurrentIs.value
        return self.motorCurrent # motor current

    # # Query motor current thread
    # def getMotorCurrentThread(self):
    #     while True:
    #         pCurrentIs=c_long()
    #         ret=epos.VCS_GetCurrentIs(self.keyHandle, self.nodeID, byref(pCurrentIs), byref(self.pErrorCode))
    #         # ret=epos.VCS_GetCurrentIsAveraged(self.keyHandle, self.nodeID, byref(pCurrentIs), byref(self.pErrorCode))
    #         self.motorCurrent = pCurrentIs.value
    #         return self.motorCurrent # motor current


    # Query analog inputs
    def getAnalogInputVoltage(self,inputNumber):
        pAnInVIs=c_long()
        ret=epos.VCS_GetAnalogInputVoltage(self.keyHandle, self.nodeID, inputNumber, byref(pAnInVIs), byref(self.pErrorCode))
        self.anInV = pAnInVIs.value
        return self.anInV # analog input voltage value

    # # Query analog inputs thread
    # def getAnalogInputVoltageThread(self,inputNumber):
    #     while True:
    #         pAnInVIs=c_long()
    #         ret=epos.VCS_GetAnalogInputVoltage(self.keyHandle, self.nodeID, inputNumber, byref(pAnInVIs), byref(self.pErrorCode))
    #         self.anInV = pAnInVIs.value
    #         return self.anInV # analog input voltage value

    
    # Query digital inputs
    def getAllDigitalInputs(self):
        pDigInIs= c_long()
        ret=epos.VCS_GetAllDigitalInputs(self.keyHandle, self.nodeID, byref(pDigInIs), byref(self.pErrorCode))
        self.digIn = pDigInIs.value
        # print(pDigInIs.value)
        return self.digIn # analog input voltage value

    # Draw Plots
    def drawPlotsStream(self):
        plt.figure(1)
        plt.show(block=False)
        while True:
            plt.subplot(221)
            plt.title('Motor Position')
            plt.plot(self.runningTime, self.motorPosition, 'b-')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (steps)')
            plt.grid(True)

            plt.subplot(222)
            plt.title('Motor Velocity')
            plt.plot(self.runningTime, self.motorVelocity, 'b-')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (steps/s)')
            plt.grid(True)

            plt.subplot(223)
            plt.title('Motor Current')
            plt.plot(self.runningTime, self.motorCurrent, 'b-')
            plt.xlabel('Time (s)')
            plt.ylabel('Current (mA)')
            plt.grid(True)

            plt.subplot(224)
            plt.title('Analog Inputs')
            plt.plot(self.runningTime, self.anInV_0, 'b-', self.runningTime, self.anInV_1, 'r-')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.grid(True)
            
            plt.draw()

            plt.pause(0.01)
        # time.sleep(1)
        
    

    def run(self):
        t_0 = time.time()
        old_pos = self.target_position
        while True:
            threadLock.acquire()
            self.motorPosition = self.getMotorPosition()
            self.motorVelocity = self.getMotorVelocity()
            self.motorCurrent = self.getMotorCurrent()
            self.anInV_0 = self.getAnalogInputVoltage(0)
            self.anInV_1 = self.getAnalogInputVoltage(1)
            self.sensorData_del_t = time.time() - t_0
            self.runningTime += self.sensorData_del_t
            if self.target_position != old_pos:
                self.MoveToPositionSpeed(self.target_position, self.speed)
            t_0 = time.time()
            self.totalTime.append(t_0)
            old_pos= self.target_position
            threadLock.release()
            

    # destructor (close connection)
    def __del__(self):
        time.sleep(1)
        epos.VCS_SetDisableState(self.keyHandle, self.nodeID, byref(self.pErrorCode)) # disable device
        epos.VCS_CloseDevice(self.keyHandle, byref(self.pErrorCode))
        
        print("Device closed")

# def multiProcessingPlotting(motor):
#     p = multiprocessing.Process(target=motor.drawPlotsStream)
#     p.start()

if __name__ == "__main__":


    # Configure desired motion profile
    acceleration = 30000 # rpm/s, up to 1e7 would be possible
    deceleration = 30000 # rpm/s
    
    EPOS1 = EPOS4USB_sensorData(b'USB0',0,1000000,500)
    EPOS1.speed = 1500
    #EPOS1.ready_motor()
    # t_motorPosition = threading.Thread(target=EPOS1.getMotorPositionThread)
    # t_motorVelocity = threading.Thread(target=EPOS1.getMotorVelocityThread)
    # t_motorCurrent = threading.Thread(target=EPOS1.getMotorCurrentThread)
    # t_analogInput_0 = threading.Thread(target=EPOS1.getAnalogInputVoltageThread,args=(0))
    # t_analogInput_1 = threading.Thread(target=EPOS1.getAnalogInputVoltage,args=(1))
    # t_digitalInputs = threading.Thread(target=EPOS1.getAllDigitalInputs)
    EPOS1.daemon = True
    EPOS1.start()
    # t_motorPosition.start()
    # t_motorVelocity.start()
    # t_motorCurrent.start()
    # t_analogInput_0.start()
    # t_analogInput_1.start()
    # t_digitalInputs.start()
    # print(EPOS1.getMotorPosition())
    # EPOS1.MoveToPositionSpeed(0, 500)
    # while True:
    #     # EPOS1.getMotorPosition()
    #     print(1/EPOS1.sensorData_del_t,EPOS1.anInV_0, EPOS1.anInV_1)
            #         create plot using matplotlib with subplots of motor position, velocity, current and analog inputs
            # make it moving window plot
        # multiProcessingPlotting(EPOS1)        

    # #del EPOS1 # close connection



    #     EPOS1.target_position = -90000 # move to position 20,000 steps at 5000 rpm/s
    #     time.sleep(0.5)

    #     EPOS1.target_position = 0 # move to position 0 steps at 2000 rpm/s
    #     time.sleep(0.5)

    # epos.VCS_SetDisableState(keyHandle, nodeID, byref(pErrorCode)) # disable device
    # epos.VCS_CloseDevice(keyHandle, byref(pErrorCode)) # close device
