import numpy as np
from math import sin, cos, radians
import cv2
class HandPoseSolver():
    def __init__(self,h5_length=66) -> None:
        self.h5_length = h5_length
        self.DH_alpha = np.array([0, -90, 0, 0, 90])
        self.DH_a = np.array([0, 0, 600, 320, 0])
        self.Initial_theta = np.array([0, 0, 0, 0, 0])  # theta角的初始值
        ###pass

    @staticmethod
    def dh_transformation(alpha, a, theta, d):  # 齐次变换矩阵公式
        alpha = radians(alpha)
        theta = radians(theta)              # 角度转弧度
        matrix = np.mat(np.zeros((4, 4)))
        matrix[0, 0] = cos(theta)
        matrix[0, 1] = -sin(theta)
        matrix[0, 3] = a
        matrix[1, 0] = sin(theta) * cos(alpha)
        matrix[1, 1] = cos(theta) * cos(alpha)
        matrix[1, 2] = -sin(alpha)
        matrix[1, 3] = -sin(alpha) * d
        matrix[2, 0] = sin(theta) * sin(alpha)
        matrix[2, 1] = cos(theta) * sin(alpha)
        matrix[2, 2] = cos(alpha)
        matrix[2, 3] = cos(alpha) * d
        matrix[3, 3] = 1
        return matrix
    
    @staticmethod
    def DOF5_matrix(DHparameter_matrix):  # 正运动学
        '''
        input: DHparameter_matrix: 5x4的矩阵, 代表DH参数
        output: 4x4的矩阵, 代表末端执行器的位姿
        '''
        DH_mat = DHparameter_matrix
        DOF5_mat = np.identity(4)
        for i in range(0, 5, 1):
            temp_mat = HandPoseSolver.dh_transformation(DH_mat[i, 0], DH_mat[i, 1], DH_mat[i, 2], DH_mat[i, 3])
            DOF5_mat = DOF5_mat * temp_mat
        return DOF5_mat

    def solver(self,theta):
        DH_alpha = np.array([0, -90, 0, 0, 90])
        DH_a = np.array([0, 0, 600, 320, 0])
        Initial_theta = np.array([0, 0, 0, 0, 0])  # theta角的初始值
        theta
        DH_theta = np.array([theta[0], theta[1],
                            theta[2], theta[3],
                            theta[4]])
        d5 = self.h5_length
        DH_d = np.array([0, 0, 0, 0, d5])
        # 把以上DH参数变为一个5x4的矩阵
        DHparameter_matrix = np.mat([[DH_alpha[0], DH_a[0], Initial_theta[0] + DH_theta[0], DH_d[0]],
                                [DH_alpha[1], DH_a[1], Initial_theta[1] + DH_theta[1], DH_d[1]],
                                [DH_alpha[2], DH_a[2], Initial_theta[2] + DH_theta[2], DH_d[2]],
                                [DH_alpha[3], DH_a[3], Initial_theta[3] + DH_theta[3], DH_d[3]],
                                [DH_alpha[4], DH_a[4], Initial_theta[4] + DH_theta[4], DH_d[4]], ])
        TT = HandPoseSolver.DOF5_matrix(DHparameter_matrix)
        return TT
    
    def Rot2RPY(self,R):
        # 使用cv2.decomposeProjectionMatrix()得到欧拉角
        _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((R, np.zeros((3, 1)))))

        roll, pitch, yaw = [np.rad2deg(angle) for angle in euler_angles]
        return [roll, pitch, yaw]


    def reverse_solver(self,oT):
        '''

        input: oT: 4x4的矩阵, 代表末端执行器的位姿
        output: np.array 5x1 [theta1, theta2, theta3, theta4, theta5] or None
        
        '''

        r11 = oT[0, 0]
        r12 = oT[0, 1]
        r13 = oT[0, 2]
        px = oT[0, 3]
        r21 = oT[1, 0]
        r22 = oT[1, 1]
        r23 = oT[1, 2]
        py = oT[1, 3]
        r31 = oT[2, 0]
        r32 = oT[2, 1]
        r33 = oT[2, 2]
        pz = oT[2, 3]

        # 求theta1
        theta1 = np.arctan2(py, px)
        s1 = sin(theta1)
        c1 = cos(theta1)

        # 求theta5
        theta5 = np.arctan2(-s1*r11+c1*r21, -s1*r12+c1*r22)
        # theta5 = theta1

        # 求theta234
        theta234 = np.arctan2(c1*r13+s1*r23, r33)
        s234 = sin(theta234)
        c234 = cos(theta234)

        # 求theta2
        d5 = self.h5_length
        a3 = 600
        a4 = 320
        A = c1*px+s1*py-s234*d5
        B = c234*d5-pz
        k = (np.square(A)+np.square(B)+np.square(a3)-np.square(a4))/(2*a3)
        theta2_1 = np.arctan2(k, np.sqrt(np.square(A)+np.square(B)-np.square(k)))-np.arctan2(A, B)
        theta2_2 = np.arctan2(k, -np.sqrt(np.square(A)+np.square(B)-np.square(k)))-np.arctan2(A, B)
        s2_1 = sin(theta2_1)
        c2_1 = cos(theta2_1)
        s2_2 = sin(theta2_2)
        c2_2 = cos(theta2_2)

        # 求theta3
        s3_1 = (B*c2_1-A*s2_1)/a4
        s3_2 = (B*c2_2-A*s2_2)/a4
        c3_1_1 = np.sqrt(1-np.square(s3_1))
        c3_1_2 = -np.sqrt(1-np.square(s3_1))
        c3_2_1 = np.sqrt(1-np.square(s3_2))
        c3_2_2 = -np.sqrt(1-np.square(s3_2))
        theta3_1_1 = np.arctan2(s3_1, c3_1_1)
        theta3_1_2 = np.arctan2(s3_1, c3_1_2)
        theta3_2_1 = np.arctan2(s3_2, c3_2_1)
        theta3_2_2 = np.arctan2(s3_2, c3_2_2)

        # 求theta4
        theta4_1_1 = theta234-theta2_1-theta3_1_1
        theta4_1_2 = theta234-theta2_1-theta3_1_2
        theta4_2_1 = theta234-theta2_2-theta3_2_1
        theta4_2_2 = theta234-theta2_2-theta3_2_2

        theta_ikine = np.mat([[theta1, theta2_1, theta3_1_1, theta4_1_1, theta5],
                            [theta1, theta2_1, theta3_1_2, theta4_1_2, theta5],
                            [theta1, theta2_2, theta3_2_1, theta4_2_1, theta5],
                            [theta1, theta2_2, theta3_2_2, theta4_2_2, theta5], ])
        
        theta_ikine = np.rad2deg(theta_ikine)  # 弧度转角度
        #print("逆解为", np.round(theta_ikine, 2))  # 输出四组解

        for i in range(0, 4):  # 判断各关节角度是否在限制范围内
            if (-45 <= theta_ikine[i, 0] <= 180 and 0 <= theta_ikine[i, 1] <= 180 and -180 <= theta_ikine[i, 2] <= 180
                and 0 <= theta_ikine[i, 3] <= 225 and -157.5 <= theta_ikine[i, 4] <= 157.5):

            # if (-180 <= theta_ikine[i, 0] <= 112.5 and 0 <= theta_ikine[i, 1] <= 180 and 0 <= theta_ikine[i, 2] <= 90
            #         and 0 <= theta_ikine[i, 3] <= 225 and -157.5 <= theta_ikine[i, 4] <= 157.5):
                #print("角度限制范围内的解为", np.round(theta_ikine[i, :], 2))
                DH_alpha = np.array([0, -90, 0, 0, 90])
                DH_a = np.array([0, 0, 600, 320, 0])
                Initial_theta = np.array([0, 0, 0, 0, 0])  # theta角的初始值
                DH_theta = np.array([theta_ikine[i, 0], theta_ikine[i, 1],
                                    theta_ikine[i, 2], theta_ikine[i, 3],
                                    theta_ikine[i, 4]])
                DH_d = np.array([0, 0, 0, 0, d5])
                # 把以上DH参数变为一个5x4的矩阵
                DHparameter_matrix = np.mat([[DH_alpha[0], DH_a[0], Initial_theta[0] + DH_theta[0], DH_d[0]],
                                        [DH_alpha[1], DH_a[1], Initial_theta[1] + DH_theta[1], DH_d[1]],
                                        [DH_alpha[2], DH_a[2], Initial_theta[2] + DH_theta[2], DH_d[2]],
                                        [DH_alpha[3], DH_a[3], Initial_theta[3] + DH_theta[3], DH_d[3]],
                                        [DH_alpha[4], DH_a[4], Initial_theta[4] + DH_theta[4], DH_d[4]], ])
                TT = HandPoseSolver.DOF5_matrix(DHparameter_matrix)
                if np.abs(TT[0,3]-oT[0,3])<1 and np.abs(TT[1,3]-oT[1,3])<1 and np.abs(TT[2,3]-oT[2,3])<1:
                    #print("解为", np.round(theta_ikine[i, :], 2))  # 输出符合条件的解
                    return np.round(theta_ikine[i, :], 2)
                continue
        return None
    

if __name__ == "__main__":
    h_solver = HandPoseSolver()
    oT = np.mat([[-1, 0, 0, 500],
                   [0, 1, 0, 0],
                   [0, 0, -1, -800],
                   [0, 0, 0, 1]])
    theta = h_solver.reverse_solver(oT)

    print(theta)

