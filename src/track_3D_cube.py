import cv2
import numpy as np
import math


def findCorners(img):

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv2.dilate(gray, kernel, iterations=1)

    gray = np.float32(img_dilation)

    height, width = img.shape[:2]
    mask1 = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(gray, mask1,(0,0),255) 
    
    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    corners = np.int0(corners)

    bottom_left = list(corners[corners[:, :, 0].argmin()][0])
    top_right = list(corners[corners[:, :, 0].argmax()][0])
    top_left = list(corners[corners[:, :, 1].argmin()][0])
    bottom_right = list(corners[corners[:, :, 1].argmax()][0])

    tag_corners = []
    tag_corners.append(top_left)
    tag_corners.append(top_right)
    tag_corners.append(bottom_left)
    tag_corners.append(bottom_right)

    return tag_corners


def homography(img):

    tag_corners = findCorners(img)

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    marker_length, marker_height = round(math.sqrt(((tag_corners[1][0] - tag_corners[0][0])**2) + ((tag_corners[1][1] - tag_corners[0][1])**2))), round(math.sqrt(((tag_corners[2][0] - tag_corners[0][0])**2) + ((tag_corners[2][1] - tag_corners[0][1])**2)))

    wc = [[0, 0], [marker_length, 0], [0, marker_height], [marker_length, marker_height]]

    A1 = np.array([(-(tag_corners[0][0])), (-(tag_corners[0][1])), -1, 0, 0, 0, ((tag_corners[0][0])*(wc[0][0])), ((tag_corners[0][1])*(wc[0][0])), ((wc[0][0]))])
    A2 = np.array([0, 0, 0, (-(tag_corners[0][0])), (-(tag_corners[0][1])), -1, ((tag_corners[0][0])*(wc[0][1])), ((tag_corners[0][1])*(wc[0][1])), ((wc[0][1]))]) 
    A3 = np.array([(-(tag_corners[1][0])), (-(tag_corners[1][1])), -1, 0, 0, 0, ((tag_corners[1][0])*(wc[1][0])), ((tag_corners[1][1])*(wc[1][0])), ((wc[1][0]))])
    A4 = np.array([0, 0, 0, (-(tag_corners[1][0])), (-(tag_corners[1][1])), -1, ((tag_corners[1][0])*(wc[1][1])), ((tag_corners[1][1])*(wc[1][1])), ((wc[1][1]))])
    A5 = np.array([(-(tag_corners[2][0])), (-(tag_corners[2][1])), -1, 0, 0, 0, ((tag_corners[2][0])*(wc[2][0])), ((tag_corners[2][1])*(wc[2][0])), ((wc[2][0]))])
    A6 = np.array([0, 0, 0, (-(tag_corners[2][0])), (-(tag_corners[2][1])), -1, ((tag_corners[2][0])*(wc[2][1])), ((tag_corners[2][1])*(wc[2][1])), ((wc[2][1]))])
    A7 = np.array([(-(tag_corners[3][0])), (-(tag_corners[3][1])), -1, 0, 0, 0, ((tag_corners[3][0])*(wc[3][0])), ((tag_corners[3][1])*(wc[3][0])), ((wc[3][0]))])
    A8 = np.array([0, 0, 0, (-(tag_corners[3][0])), (-(tag_corners[3][1])), -1, ((tag_corners[3][0])*(wc[3][1])), ((tag_corners[3][1])*(wc[3][1])), ((wc[3][1]))])

    A = np.array([A1, A2, A3, A4, A5, A6, A7, A8])

    _, __, V_T = np.linalg.svd(A)

    H = np.divide((V_T[-1]), (V_T[-1])[-1])
    H = np.reshape(H, (3,3))

    return H, marker_height, marker_length



def P_matrix(frame):

    file = open("./data/kmatrix.csv")
    K = np.loadtxt(file, delimiter=",")

    H, _, __ = homography(frame)

    l = 1/(((np.linalg.norm(np.matmul((np.linalg.inv(K)), H[:, 0]))) + (np.linalg.norm(np.matmul((np.linalg.inv(K)), H[:, 1]))))/2)

    B_tilda = np.matmul((np.linalg.inv(K)), np.linalg.inv(H))

    if np.linalg.det(B_tilda) < 0:
        B = (-1) * (B_tilda)
    else:
        B = B_tilda

    r1 = l * (B[:, 0])
    r2 = l * (B[:, 1])
    r3 = np.cross(r1, r2)
    t = l * (B[:, 2])

    Rt = np.c_[r1, r2, r3, t]

    P = np.asarray(np.matmul(K, Rt))

    return P


def cube(frame):

    tag_corners = findCorners(frame)
    _, marker_height, marker_length = homography(frame)
    P = P_matrix(frame)

    w_cord_1 = (np.array([0, 0, -marker_height, 1]).T)
    w_cord_2 = (np.array([0, marker_length, -marker_height, 1]).T)
    w_cord_3 = (np.array([marker_height, 0, -marker_height, 1]).T)
    w_cord_4 = (np.array([marker_height, marker_length, -marker_height, 1]).T)

    cam_cord_1 = np.matmul(P, w_cord_1)
    cam_cord_1 = np.divide(cam_cord_1, cam_cord_1[-1])
    cam_cord_2 = np.matmul(P, w_cord_2)
    cam_cord_2 = np.divide(cam_cord_2, cam_cord_2[-1])
    cam_cord_3 = np.matmul(P, w_cord_3)
    cam_cord_3 = np.divide(cam_cord_3, cam_cord_3[-1])
    cam_cord_4 = np.matmul(P, w_cord_4)
    cam_cord_4 = np.divide(cam_cord_4, cam_cord_4[-1])

    point_1 = (np.abs(round(cam_cord_1[0])), np.abs(round(cam_cord_1[1])))
    point_2 = (np.abs(round(cam_cord_3[0])), np.abs(round(cam_cord_3[1])))
    point_3 = (np.abs(round(cam_cord_2[0])), np.abs(round(cam_cord_2[1])))
    point_4 = (np.abs(round(cam_cord_4[0])), np.abs(round(cam_cord_4[1])))

    corner_1 = (tag_corners[0][0], tag_corners[0][1])
    corner_2 = (tag_corners[1][0], tag_corners[1][1])
    corner_3 = (tag_corners[2][0], tag_corners[2][1])
    corner_4 = (tag_corners[3][0], tag_corners[3][1])

    frame = cv2.line(frame, corner_1, corner_2, (0,255,0), 5)
    frame = cv2.line(frame, corner_1, corner_3, (0,255,0), 5)
    frame = cv2.line(frame, corner_3, corner_4, (0,255,0), 5)
    frame = cv2.line(frame, corner_2, corner_4, (0,255,0), 5)

    frame = cv2.line(frame, corner_1, point_1, (0,255,0), 5)
    frame = cv2.line(frame, corner_2, point_2, (0,255,0), 5)
    frame = cv2.line(frame, corner_3, point_3, (0,255,0), 5)
    frame = cv2.line(frame, corner_4, point_4, (0,255,0), 5)

    frame = cv2.line(frame, point_1, point_2, (0,255,0), 5)
    frame = cv2.line(frame, point_1, point_3, (0,255,0), 5)
    frame = cv2.line(frame, point_3, point_4, (0,255,0), 5)
    frame = cv2.line(frame, point_2, point_4, (0,255,0), 5)

    return frame, marker_height, marker_length


def track_cube(video):

    cap = cv2.VideoCapture(video)
    iter = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == False:
            break

        final, marker_height, marker_length = cube(frame)

        if (100 < marker_height < 200) and (100 < marker_length < 200):

            cv2.imshow('dst', final)
            cv2.waitKey(1)

        print(iter)
        iter += 1

    cap.release()

    cv2.destroyAllWindows()
