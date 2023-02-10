import cv2
import numpy as np
import os

from crestereo.crestereo import CREStereo, _log_time_usage


def read_stereo_pairs_from_folder(folder_path, extension='.png'):
	stereo_pair_paths = []
	for filename in os.listdir(folder_path):
		if extension in filename:
			if filename[-8:-4] == 'left':
				left_image_name = filename
				right_image_name = filename[:-8] + \
					filename[-8:-4].replace('left', 'right')+extension
				if os.path.exists(folder_path+'/'+right_image_name):
					stereo_pair_paths.append(
						(folder_path+'/'+left_image_name, folder_path+'/'+right_image_name))
	return stereo_pair_paths


if __name__ == '__main__':

	input_folder_path = "/home/aboggaram/data/Octiva/stereo_test_raw_images"
	# Load images
	stereo_pair_paths = read_stereo_pairs_from_folder(input_folder_path)

	# Model Selection options (not all options supported together)
	iters = 10            # Lower iterations are faster, but will lower detail. 
						# Options: 2, 5, 10, 20 

	shape = (320, 480)   # Input resolution. 
						# Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

	version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
						# Options: "init", "combined"


	# Initialize model
	model_path = f'models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx'
	# trt_engine_path = '/home/aboggaram/models/TensorrtExecutionProvider_TRTKernel_graph_next_onnx::Neg_8633_18146894986420854276_1_0_fp16.engine'
	with _log_time_usage("Model Initialization and Optimization: "):
		depth_estimator = CREStereo(model_path)

	output_folder = '/home/aboggaram/data/Octiva/iunu_stereo_test_output'

	for i in range(len(stereo_pair_paths)):

		left_img_path, right_img_path = stereo_pair_paths[i]
		left_image_name = left_img_path.split('/')[-1]
		left_img = cv2.imread(left_img_path)
		right_img = cv2.imread(right_img_path)

		with _log_time_usage("depth inference 1 pair: "):
			disparity_map = depth_estimator(left_img, right_img)

		color_disparity = depth_estimator.draw_disparity()

		combined_image = np.hstack((left_img, color_disparity))
		output_image_path = output_folder+'/'+left_image_name
		cv2.imwrite(output_image_path, combined_image)
		cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
		cv2.imshow("Estimated disparity", combined_image)

		# Press key q to stop
		pressed_key = cv2.waitKey(1)
		if pressed_key == ord('q'):
			break
		elif pressed_key == 32:
			continue
		else:
			cv2.waitKey(delay=100)
	cv2.destroyAllWindows()
	del depth_estimator

 