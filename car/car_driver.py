from __future__ import print_function
import RPi.GPIO as io
io.setmode(io.BCM)
import time


# FIXME: enable PWM
class Wheel():
	def __init__(self, in1, in2):
		self.in1 = in1
		self.in2 = in2

	def setup(self):
		io.setup(self.in1, io.OUT)
		io.setup(self.in2, io.OUT)
		# motor1 = io.PWM(4,100)
		# motor1.start(0)
		# motor1.ChangeDutyCycle(0)

	def forward(self, **kwargs):
		io.output(self.in1, True)
		io.output(self.in2, False)

	def reverse(self, **kwargs):
		io.output(self.in1, False)
		io.output(self.in2, True)

	def stop(self, **kwargs):
		io.output(self.in1, False)
		io.output(self.in2, False)


class Driver():
	def __init__(self):
		# self.front_left = Wheel(4, 17) # physical pins 7 11
		self.front_left = Wheel(17, 27) # physical pins 11 13
		self.front_right = Wheel(23, 24) # physical pins 16 18
		self.back_left = Wheel(5, 6) # physical pins ? 29 31
		self.back_right = Wheel(12, 16) # physical pins ? 32 36

		self.state = 'stop'
		self.setup()

	def setup(self):
		self.front_left.setup()
		self.front_right.setup()
		self.back_left.setup()
		self.back_right.setup()

	def forward(self, **kwargs):
		self.front_left.forward(**kwargs)
		self.front_right.forward(**kwargs)
		self.back_right.forward(**kwargs)
		self.back_left.forward(**kwargs)
		self.state = 'forward'

	def reverse(self, **kwargs):
		self.front_left.reverse(**kwargs)
		self.front_right.reverse(**kwargs)
		self.back_right.reverse(**kwargs)
		self.back_left.reverse(**kwargs)
		self.state = 'reverse'

	def turn_left(self, **kwargs):
		self.front_left.stop(**kwargs)
		self.front_right.forward(**kwargs)
		self.back_right.forward(**kwargs)
		self.back_left.stop(**kwargs)
		self.state = 'left'
		if "delay" in kwargs:
			time.sleep(kwargs['delay'])
			if "callback" in kwargs:
				kwargs['callback']()
			else:
				self.stop()

	def turn_right(self, **kwargs):
		self.front_left.forward(**kwargs)
		self.front_right.stop(**kwargs)
		self.back_right.stop(**kwargs)
		self.back_left.forward(**kwargs)
		self.state = 'right'
		if "delay" in kwargs:
			time.sleep(kwargs['delay'])
			if "callback" in kwargs:
				kwargs['callback']()
			else:
				self.stop()

	def stop(self, **kwargs):
		self.front_left.stop(**kwargs)
		self.front_right.stop(**kwargs)
		self.back_right.stop(**kwargs)
		self.back_left.stop(**kwargs)
		self.state = 'stop'

	def __del__(self):
		self.stop()
		io.cleanup()

	def cleanup(self):
		self.stop()
		io.cleanup()


if __name__ == '__main__':
	car = Driver()

	try:

		time.sleep(0.5)
		car.stop()
		# time.sleep(0.5)
		car.forward()
		time.sleep(1)
		car.reverse()
		time.sleep(1)
		car.turn_left()
		time.sleep(1)
		car.turn_right()
		time.sleep(1)
		car.stop()
	except:
		car.cleanup()
	finally:
		car.cleanup()