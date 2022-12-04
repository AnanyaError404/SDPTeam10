
import machine
from machine import Pin
import utime

#inputs from yoga mat that need to be sent to Open CV. Open cv will interpret if the selected pose requires both pressure sensors actuated,
#only 1 presure sensor, or 0 pressure sensors.
left_pressure_sensor = Pin(21, Pin.IN, Pin.PULL_DOWN)
right_pressure_sensor = Pin(20, Pin.IN, Pin.PULL_DOWN)

#outputs to LED face on stickman
led_red = machine.Pin(9, machine.Pin.OUT)
led_yellow = machine.Pin(10, machine.Pin.OUT)
led_green = machine.Pin(14, machine.Pin.OUT)


#Open cv should only output these 3 variables to the Raspberry pi, whether the posture is good, okay, or bad. 
good_posture = 1
okay_posture = 0
bad_posture = 0
 
 
while True:
    
    #These variables need to be sent to the computer opencv program
    if left_pressure_sensor.value() == 1:
        print("Left Pressure Sensor Actuated")
    if left_pressure_sensor.value() == 0:
        print("Left Pressure Sensor Unactuated")
        
    if right_pressure_sensor.value() == 1:
        print("Right Pressure Sensor Actuated")
    if right_pressure_sensor.value() == 0:
        print("Right Pressure Sensor Unactuated")
    
    
    #These are the outputs to the LED face
    if good_posture:
        led_green.value(1)
        led_yellow.value(0)
        led_red.value(0)
    if okay_posture:
        led_green.value(0)
        led_yellow.value(1)
        led_red.value(0)
    if bad_posture:
        led_green.value(0)
        led_yellow.value(0)
        led_red.value(1)
    
    utime.sleep(1)
    
