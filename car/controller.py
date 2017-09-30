import urllib2
# import pyping
import car_driver
import os
import json
import time
# import pprint
import argparse

# lacking port number
IP_ADDRESS = '192.168.0.'
car = None

TURN_DELAY = 0.7

def response_label(car, response):
	global TURN_DELAY
	label = response.get('label')

	if label == 'left':
		if car.state != 'stop':
			car.turn_left()
		# car.turn_left(delay=TURN_DELAY)
	elif label == 'right':
		# car.turn_right(delay=TURN_DELAY)
		if car.state != 'stop':
			car.turn_right()
	elif label == 'double_blink':
		if car.state == 'forward':
			car.stop()
		else:
			car.forward()
	# keep the state
	else:
		if car.state == 'stop':
			car.stop()
		else:
			car.forward()
	#
	# if label == 'up':
	# 	if car.state == 'stop':
	# 		car.stop()
	# 	else:
	# 		car.forward()
	#
	# if label == 'center':
	# 	if car.state == 'stop':
	# 		car.stop()
	# 	else:
	# 		car.forward()
	# if label == 'down':
	# 	car.reverse()
	# if label == 'clockwise':
	# 	pass
	# if label == 'counter_clockwise':
	# 	pass


def control_loop():
	global IP_ADDRESS, car
	try:
		url = "http://{}/inference".format(IP_ADDRESS)
		car = car_driver.Driver()
		time.sleep(0.5)
		interval = 0
		while True:
			interval = time.time()
			response_json = urllib2.urlopen(url).read()
			# print response_json
			response = json.loads(response_json)
			print "time delay: {} - {}".format(time.time() - interval, response['label'])
			# print response['label']
			# print 'Prediction'
			# pprint.pprint(response)
			if response:
				response_label(car, response)

			# time.sleep(0.1)

	except KeyboardInterrupt:
		print 'keyboard interrupt'
		car.stop()
	except Exception as e:
		print e
	finally:
		print 'clean up'
		car.stop()
		car.cleanup()
		exit()

def get_url():
	global IP_ADDRESS, car
	for i in range(100, 255):
		url = '{}{}'.format(IP_ADDRESS, i)
		try:
			# status = pyping.ping(url).ret_code
			# if pass
			# status = True if os.system("ping -c 1 {}".format(url)) is 0 else False
			status = urllib2.urlopen("http://{}/ping".format(url)).read()
			print status

			if status and status != '':
				print 'Pinging {} success'.format(url)
				IP_ADDRESS = url
				break
		except KeyboardInterrupt:
			print 'Keyboard interrupt'
			break
		except:
			print 'Passing ping {}'.format(url)


def main():
	# get_url()
	global IP_ADDRESS
	parser = argparse.ArgumentParser()
	parser.add_argument('--ip', help='Epoch Ip address of headset', type=str, required=True)
	args = parser.parse_args()

	IP_ADDRESS = args.ip
	control_loop()

if __name__ == '__main__':
	main()