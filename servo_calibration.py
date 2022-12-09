#!/usr/bin/python

'''
This program is used to calibrate 16 servos controlled by PCA9685.
'''

import time
import math
import smbus
import sys
import select
import termios
import tty

# Global variable for number of servos, min pulse us, center pulse us, max pulse us, min step us, max step us
numServos = 16
minPulse = 500
centerPulse = 1500
maxPulse = 2500
minStep = 4
maxStep = 40

msg = f'''
Servo Calibration Module for {numServos} Servos.

Servo parameters:
Range: 180 deg, 500us ~ 2500us
Dead band: 4us
Freq: 50Hz

Enter one of the following options:
-----------------------------
quit: stop and quit the program
oneServo: Move one servo manually, the servo is first set to the center position
allServos: Move all servo's manually together, all others will be commanded to their center position
Keyboard commands for One Servo Control
---------------------------
   q                y
            f   g       j   k
    z   x       b   n   m
  q: Quit current command mode and go back to Option Select
  z: Command servo min value {minPulse}us
  y: Command servo center value {centerPulse}us
  x: Command servo max value {maxPulse}us
  f: Manually decrease servo command value by {maxStep}us
  g: Manually decrease servo command value by {minStep}us
  j: Manually increase servo command value by {minStep}us
  k: Manually increase servo command value by {maxStep}us
  b: Save new min command value
  n: Save new center command value
  m: Save new max command value
  anything else : Prompt again for command
CTRL-C to quit
'''

# Dictionary with anonomous helper functions to execute key commands
keyDict = {
    'q': None,
    'z': lambda x: x.set_value(x.pulse_min()),
    'y': lambda x: x.set_value(x.pulse_center()),
    'x': lambda x: x.set_value(x.pulse_max()),
    'f': lambda x: x.set_value(x.pulse_value()-maxStep),
    'g': lambda x: x.set_value(x.pulse_value()-minStep),
    'j': lambda x: x.set_value(x.pulse_value()+minStep),
    'k': lambda x: x.set_value(x.pulse_value()+maxStep),
    'b': lambda x: x.set_min(x.pulse_value()),
    'n': lambda x: x.set_center(x.pulse_value()),
    'm': lambda x: x.set_max(x.pulse_value()),
}

validCmds = ['quit', 'oneServo', 'allServos']


class PCA9685:
    '''
    PCA9685 16-Channel PWM Servo Driver
    '''

    # Registers/etc.
    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address=0x40, debug=False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        if (self.debug):
            print("Reseting PCA9685")
        self.write(self.__MODE1, 0x00)

    def write(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        self.bus.write_byte_data(self.address, reg, value)
        if (self.debug):
            print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))

    def read(self, reg):
        "Read an unsigned byte from the I2C device"
        result = self.bus.read_byte_data(self.address, reg)
        if (self.debug):
            print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" %
                  (self.address, result & 0xFF, reg))
        return result

    def setPWMFreq(self, freq):
        "Sets the PWM frequency"
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if (self.debug):
            print("Setting PWM frequency to %d Hz" % freq)
            print("Estimated pre-scale: %d" % prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if (self.debug):
            print("Final pre-scale: %d" % prescale)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L+4*channel, on & 0xFF)
        self.write(self.__LED0_ON_H+4*channel, on >> 8)
        self.write(self.__LED0_OFF_L+4*channel, off & 0xFF)
        self.write(self.__LED0_OFF_H+4*channel, off >> 8)
        if (self.debug):
            print("channel: %d  LED_ON: %d LED_OFF: %d" %
                  (channel, on, off))

    def setServoPulse(self, channel, pulse):
        "Sets the Servo Pulse,The PWM frequency must be 50HZ"
        # PWM frequency is 50HZ,the period is 20000us
        pwm = int(pulse*4096/20000)
        self.setPWM(channel, 0, pwm)
        print('PWM:', pwm)


class ServoData():
    '''
    ServoData Class encapsulates a servo 
    Servo has a min, center and max pulse value, and is commanded by a value between min pulse and max pulse.
    This coorsponds to the duty cycle in a 12 bit pwm cycle. Nominally, a servo is commanded with pulses of
    0.5 to 2.5 ms in a 20 ms cycle, with 1.5 ms being the value for center position. 
    '''

    def __init__(self, id=1, min_pulse=minPulse, center_pulse=centerPulse, max_pulse=maxPulse):
        self.__value = center_pulse
        self.__center = center_pulse
        self.__min = min_pulse
        self.__max = max_pulse
        self.id = id

    def set_value(self, value_in):
        '''
        Set Servo value
        Input: Value between min pulse and max pulse
        '''
        if value_in not in range(minPulse, maxPulse+1):
            print(f'Servo value not in range [{minPulse},{maxPulse}]')
        else:
            self.__value = value_in

    def set_center(self, center_val):
        '''
        Set Servo center value
        Input: Value between 500 and 2500
        '''
        if center_val not in range(minPulse, maxPulse+1):
            print(f'Servo value not in range [{minPulse},{maxPulse}]')
        else:
            self.__center = center_val
            print('Servo %2i center set to %4i' % (self.id+1, center_val))

    def set_max(self, max_val):
        '''
        Set Servo max value
        Input: Value between 500 and 2500
        '''
        if max_val not in range(minPulse, maxPulse+1):
            print(f'Servo value not in range [{minPulse},{maxPulse}]')
        else:
            self.__max = max_val
            print('Servo %2i max set to %4i' % (self.id+1, max_val))

    def set_min(self, min_val):
        '''
        Set Servo min value
        Input: Value between 500 and 2500
        '''
        if min_val not in range(minPulse, maxPulse+1):
            print(f'Servo value not in range [{minPulse},{maxPulse}]')
        else:
            self.__min = min_val
            print('Servo %2i min set to %4i' % (self.id+1, min_val))

    def pulse_value(self):
        return self.__value

    def pulse_min(self):
        return self.__min

    def pulse_center(self):
        return self.__center

    def pulse_max(self):
        return self.__max

    def __pulse2pwm(self, pulse):
        return int(pulse*4096/20000)

    def pwm_min(self):
        return self.__pulse2pwm(self.__min)

    def pwm_center(self):
        return self.__pulse2pwm(self.__center)

    def pwm_max(self):
        return self.__pulse2pwm(self.__max)


settings = termios.tcgetattr(sys.stdin)


def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__ == '__main__':

    pwm = PCA9685(0x40, debug=False)
    pwm.setPWMFreq(50)

    servos = {i: ServoData(i) for i in range(numServos)}

    while True:

        print(msg)
        userInput = input("Command?: ")

        if userInput not in validCmds:
            print('Valid command not entered, try again...')
        else:
            if userInput == 'quit':
                print("Ending program...")
                print('Final Servo Values')
                print('--------------------')
                for i in range(numServos):
                    print('Servo %2i PWM:   Min: %4i,   Center: %4i,   Max: %4i' % (
                        i, servos[i].pwm_min(), servos[i].pwm_center(), servos[i].pwm_max()))
                break

            elif userInput == 'oneServo':
                # First get servo number to command
                nSrv = -1
                while (1):
                    userInput = int(
                        input('Which servo to control? Enter a number 1 through 12: '))

                    if userInput not in range(1, numServos+1):
                        print("Invalid servo number entered, try again")
                    else:
                        nSrv = userInput - 1
                        break

                # Reset the servo to center value, and send command
                pwm.setServoPulse(nSrv, servos[nSrv].pulse_center())

                # Loop and act on user command
                print('Enter command, q to go back to option select: ')
                while (1):

                    userInput = getKey()

                    if userInput == 'q':
                        break
                    elif userInput not in keyDict:
                        print('Key not in valid key commands, try again')
                    else:
                        keyDict[userInput](servos[nSrv])
                        print('Servo %2i cmd: %i' %
                              (nSrv, servos[nSrv].pulse_value()))
                        pwm.setServoPulse(nSrv, servos[nSrv].pulse_value())

            elif userInput == 'allServos':
                # Reset all servos to center value, and send command
                for i in range(numServos):
                    pwm.setServoPulse(i, servos[i].pulse_center())

                print('Enter command, q to go back to option select: ')
                while (1):

                    userInput = getKey()

                    if userInput == 'q':
                        break
                    elif userInput not in keyDict:
                        print('Key not in valid key commands, try again')
                    elif userInput in ('b', 'n', 'm'):
                        print(
                            'Saving values not supported in all servo control mode')
                    else:
                        for s in servos.values():
                            keyDict[userInput](s)
                            pwm.setServoPulse(s.id, s.pulse_value())
                        print('All Servos Commanded')
