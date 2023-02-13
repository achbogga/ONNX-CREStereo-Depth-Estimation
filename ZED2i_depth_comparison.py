import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
from crestereo.crestereo import CameraConfig, CREStereo, _log_time_usage
from copy import deepcopy

def main():
	# Create a Camera object
	zed = sl.Camera()

	# Create a InitParameters object and set configuration parameters
	init_params = sl.InitParameters()
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
	init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
	init_params.camera_resolution = sl.RESOLUTION.HD720

	# Model Selection options (not all options supported together)
	iters = 2            # Lower iterations are faster, but will lower detail. 
						# Options: 2, 5, 10, 20 

	shape = (720, 1280)   # Input resolution. 
						# Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

	version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
						# Options: "init", "combined"

	# Open the camera
	err = zed.open(init_params)
	if err != sl.ERROR_CODE.SUCCESS:
		exit(1)

	calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
	# Focal length of the left eye in pixels
	focal_left_x = calibration_params.left_cam.fx
	stereo_baseline = calibration_params.get_camera_baseline()
	max_dist = 1.5
	print ('focal_left_x in px: ', focal_left_x)
	print ('stereo_baseline: ', stereo_baseline)
	zed2i_camera_config = CameraConfig(stereo_baseline, focal_left_x)


	# Initialize model
	model_path = f'models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx'

	with _log_time_usage("Model Initialization and Optimization: "):
		depth_estimator = CREStereo(model_path = model_path, camera_config = zed2i_camera_config, max_dist=max_dist)

	# Create and set RuntimeParameters after opening the camera
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Use STANDARD or FILL sensing modes
	# Setting the depth confidence parameters
	runtime_parameters.confidence_threshold = 100
	runtime_parameters.textureness_confidence_threshold = 100

	# Capture 150 images and depth, then stop
	i = 0
	left_image = sl.Mat()
	right_image = sl.Mat()
	depth = sl.Mat()
	point_cloud = sl.Mat()

	mirror_ref = sl.Transform()
	mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
	tr_np = mirror_ref.m

	while i < 10:
		# A new image is available if grab() returns SUCCESS
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
			# Retrieve left image
			zed.retrieve_image(left_image, sl.VIEW.LEFT)
			zed.retrieve_image(right_image, sl.VIEW.RIGHT)
			# Use get_data() to get the numpy array
			left_image_array = left_image.get_data()
			right_image_array = right_image.get_data()


			with _log_time_usage(' inference time (s) for 720p pair: '):
				disparity_map = depth_estimator(left_img=left_image_array, right_img=right_image_array)
			depth_map_from_crestereo = deepcopy(depth_estimator.depth_map)

			color_depth_crestereo = deepcopy(depth_estimator.draw_depth())

			# Retrieve depth map. Depth is aligned on the left image
			zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

			depth_map_from_zed = deepcopy(depth.get_data())

			depth_estimator.depth_map = depth_map_from_zed
			color_depth_map_zed = deepcopy(depth_estimator.draw_depth())
			combined_image = np.hstack((color_depth_map_zed, color_depth_crestereo))
			combined_image = cv2.resize(combined_image, (1920, 1080))
			cv2.imshow('zed_cre_depth_comparison', combined_image)
			depth_map_from_crestereo[np.isnan(depth_map_from_crestereo)] = 0
			depth_map_from_crestereo[np.isinf(depth_map_from_crestereo)] = 0
			depth_map_from_zed[np.isinf(depth_map_from_zed)] = 0
			depth_map_from_zed[np.isnan(depth_map_from_zed)] = 0


			print ('Depth Maps MSE: ', (np.square(depth_map_from_crestereo - depth_map_from_zed)).mean())


			# # Retrieve colored point cloud. Point cloud is aligned on the left image.
			# zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

			# # Get and print distance value in mm at the center of the image
			# # We measure the distance camera - object using Euclidean distance
			# x = round(left_image.get_width() / 2)
			# y = round(right_image.get_height() / 2)
			# err, point_cloud_value = point_cloud.get_value(x, y)

			# distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
			# 					 point_cloud_value[1] * point_cloud_value[1] +
			# 					 point_cloud_value[2] * point_cloud_value[2])

			# point_cloud_np = point_cloud.get_data()
			# point_cloud_np.dot(tr_np)

			# if not np.isnan(distance) and not np.isinf(distance):
			# 	print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(x, y, distance), end="\r")
			# 	# Increment the loop
			# 	i = i + 1
			# else:
			# 	print("Can't estimate distance at this position.")
			# 	print("Your camera is probably too close to the scene, please move it backwards.\n")
			sys.stdout.flush()
			i+=1
			# Press key q to stop
			pressed_key = cv2.waitKey(1)
			if pressed_key == ord('q'):
				break
			elif pressed_key == 32:
				continue
			else:
				cv2.waitKey(delay=5000)
		else:
			print ('camera not ready yet')
			pass
	cv2.destroyAllWindows()

	# Close the camera
	zed.close()

if __name__ == "__main__":
	main()
