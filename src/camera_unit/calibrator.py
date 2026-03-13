import cv2
import numpy as np
import os

class StereoCalibrator:
    def __init__(self, config):
        self.pattern_size = tuple(config['calibration']['pattern_size'])
        self.square_size = config['calibration']['square_size_mm']
        
        # 準備棋盤格在現實世界中的 3D 座標 (Z=0)
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # 儲存所有影像的點
        self.objpoints = [] # 3D points in real world space
        self.imgpoints_l = [] # 2D points in image plane (Left)
        self.imgpoints_r = [] # 2D points in image plane (Right)

    def add_corners(self, img_l, img_r):
        """偵測並記錄一對影像的角點"""
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.pattern_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.pattern_size, None)

        if ret_l and ret_r:
            # 精細化角點座標
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

            self.objpoints.append(self.objp)
            self.imgpoints_l.append(corners_l)
            self.imgpoints_r.append(corners_r)
            return True, (corners_l, corners_r)
        return False, None

    def calibrate(self, img_shape):
        """執行雙目校正"""
        # 1. 先分別計算單機內參 (Camera Matrix & Distortion)
        ret_l, K_l, D_l, R_l, T_l = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None, None)
        ret_r, K_r, D_r, R_r, T_r = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None, None)

        # 2. 進行雙目校正得到 R (旋轉矩陣) 與 T (平移向量)
        flags = cv2.CALIB_FIX_INTRINSIC # 固定內參，只求相對位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            K_l, D_l, K_r, D_r, img_shape,
            criteria=criteria, flags=flags
        )
        
        return {
            "K_l": K_l, "D_l": D_l,
            "K_r": K_r, "D_r": D_r,
            "R": R, "T": T
        }

    def save_results(self, data, path):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        for k, v in data.items():
            fs.write(k, v)
        fs.release()