class ProcessingParams:
	def __init__(self, frame_resized, yolo_model, window_scale_factor, car_fix, car_fix2, car_back_img,
				 car_back_imgS, car_front_imgS, car_front_img, stop_img, mtx, dist, th_sobelx, th_sobely, th_mag, th_dir, th_h,
				 th_l, th_s, left_line, right_line, focal_length_px, vehicle_height_m, moto_back_img,
				 moto_back_imgS, car_fix_curve_left, car_fix_curve_right, car_fix_move, car_back_imgM, car_front_imgM, moto_back_imgM,
				 car_fix2_move, car_fix_curve_left_move, car_fix_curve_right_move,
				 truck_back_img, truck_back_imgM, truck_back_imgS):
		self.frame_resized = frame_resized
		self.yolo_model = yolo_model
		self.window_scale_factor = window_scale_factor
		self.car_fix = car_fix
		self.car_fix2 = car_fix2
		self.car_back_img = car_back_img
		self.car_back_imgS = car_back_imgS
		self.car_front_imgS = car_front_imgS
		self.car_front_img = car_front_img
		self.stop_img = stop_img
		self.mtx = mtx
		self.dist = dist
		self.th_sobelx = th_sobelx
		self.th_sobely = th_sobely
		self.th_mag = th_mag
		self.th_dir = th_dir
		self.th_h = th_h
		self.th_l = th_l
		self.th_s = th_s
		self.left_line = left_line
		self.right_line = right_line
		self.focal_length_px = focal_length_px
		self.vehicle_height_m = vehicle_height_m
		self.moto_back_img = moto_back_img
		self.moto_back_imgS = moto_back_imgS
		self.car_fix_curve_left = car_fix_curve_left
		self.car_fix_curve_right = car_fix_curve_right
		self.car_fix_move = car_fix_move
		self.car_back_imgM = car_back_imgM
		self.car_front_imgM = car_front_imgM
		self.moto_back_imgM = moto_back_imgM
		self.car_fix2_move = car_fix2_move
		self.car_fix_curve_left_move = car_fix_curve_left_move
		self.car_fix_curve_right_move = car_fix_curve_right_move
		self.truck_back_img = truck_back_img
		self.truck_back_imgM = truck_back_imgM
		self.truck_back_imgS = truck_back_imgS