#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:27:16 2019

@author: pi
"""

import socket
import numpy as np
import time

COMMAND_LENGTH_IN_BYTES = 2


class Sessantaquattro():
    def __init__(self, trigger=0, wifi_range=0, hpfilter=1, resolution=0, mode=0, nch=3, fsamp=2, buffsize=200):

        self._ip = "0.0.0.0"
        self._port = 45454
        self._sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._buffsize = buffsize
        self._trig = trigger
        self._ext = wifi_range
        self._hpf = hpfilter
        self._hres = resolution
        self._mode = mode
        self._nch = nch
        self._fsamp = fsamp
        number_of_channels = None
        sample_frequency = None
        bytes_in_sample = None

        if nch == 0:
            if mode == 1:
                number_of_channels = 8
            else:
                number_of_channels = 12
        elif nch == 1:
            if mode == 1:
                number_of_channels = 12
            else:
                number_of_channels = 20
        elif nch == 2:
            if mode == 1:
                number_of_channels = 20
            else:
                number_of_channels = 36
        elif nch == 3:
            if mode == 1:
                number_of_channels = 36
            else:
                number_of_channels = 68
        else:
            raise Exception("wrong value for nch. Got: {}".format(nch))

        if fsamp == 0:
            if mode == 3:
                sample_frequency = 2000
            else:
                sample_frequency = 500
        elif fsamp == 1:
            if mode == 3:
                sample_frequency = 4000
            else:
                sample_frequency = 1000
        elif fsamp == 2:
            if mode == 3:
                sample_frequency = 8000
            else:
                sample_frequency = 2000
        elif fsamp == 3:
            if mode == 3:
                sample_frequency = 16000
            else:
                sample_frequency = 4000
        else:
            raise Exception("wrong value for fsamp. Got: {}".format(fsamp))

        if resolution == 1:
            bytes_in_sample = 3
        else:
            bytes_in_sample = 2

        if number_of_channels is None or sample_frequency is None or bytes_in_sample is None:
            raise Exception("Could not set number_of_channels and/or  and/or bytes_in_sample")

        self._buffer_values = np.zeros((self._buffsize, number_of_channels), dtype=np.int32)
        self._one_sample_values = np.zeros((number_of_channels,), dtype=np.int32)
        self._number_of_channels = number_of_channels
        self._sample_frequency = sample_frequency
        self._bytes_in_sample = bytes_in_sample
        print(nch)

    def create_bin_command(self, go, rec=0, getset=0):
        command = 0
        command = command + go
        command = command + rec * 2
        command = command + self._trig * 4
        command = command + self._ext * 16
        command = command + self._hpf * 64
        command = command + self._hres * 128
        command = command + self._mode * 256
        command = command + self._nch * 2048
        command = command + self._fsamp * 8192
        command = command + getset * 32768

        return int(command).to_bytes(COMMAND_LENGTH_IN_BYTES, byteorder="big")

    def connect_to_sq(self):
        start_command = self.create_bin_command(go=1)
        print("Server binding")
        self._sq_socket.bind((self._ip, self._port))
        print("Server listening")
        self._sq_socket.listen(1)
        self._conn, addr = self._sq_socket.accept()
        print("Connection address: {}".format(addr))
        self._conn.send(start_command)

    def disconnect_from_sq(self):
        if self._conn is not None:
            stop_command = self.create_bin_command(go=0)
            self._conn.send(stop_command)
            self._conn.shutdown(2)
            self._conn.close()
        else:
            raise Exception("Can't disconnect from because the connection is not established")

    def read_emg(self):
        # bdata = np.zeros((self._buffsize * self._number_of_channels), dtype=np.int32)
        # emgarray = np.zeros((self._buffsize, self._number_of_channels), dtype=np.int32)
        buffer_size = self._number_of_channels * self._bytes_in_sample * self._buffsize
        raw_data_stream = self._conn.recv(buffer_size, socket.MSG_WAITALL)

        # Read the data row by row
        bdata = np.frombuffer(buffer=raw_data_stream, dtype='>h', count=self._buffsize * self._number_of_channels)
        emgarray = np.reshape(bdata, [self._buffsize, self._number_of_channels], order='C')

        return emgarray, raw_data_stream
