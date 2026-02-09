import numpy as np

class HandControl():

    def __init__(self,bitrate=50000):
      
        self.R = np.array([[0,-1,0],
        [-0.707,0,-0.707],
        [0.707,0,-0.707]]).T
        self.hand_eye_mat = np.eye(4)
        self.trans = np.array([80,0,50])
        self.hand_eye_mat[:3,:3] = self.R
        self.hand_eye_mat[:3,3] = self.trans
        print(self.hand_eye_mat)

    def transform_eye2hand(self,position):
        p_hand = self.hand_eye_mat @ np.array([[position[0]],[position[1]],[position[2]],[1]])

        return p_hand[:3].T


if __name__ == '__main__':
    hand = HandControl()
    p_in_hand = hand.transform_eye2hand([0,0,1000])
    print(p_in_hand)

           
