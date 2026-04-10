# Write your code here :-)
import gpiozero
import time

from gpiozero.pins.mock import MockFactory

gpiozero.Device.pin_factory = MockFactory()
from gpiozero import LED
led = LED(17)
led.on()


gpiozero.Device.pin_factory = MockFactory()

# Create a list of all GPIO pins
pins = [LED(i) for i in range(28)]

# Print the state of all GPIOs
while True:
    for pin in pins:
        print(f"GPIO pin {pin.pin.number}: {pin.is_lit}")
