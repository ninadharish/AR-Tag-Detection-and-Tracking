import cv2
import numpy as np
import math


def findCorners(img):

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
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


    if (100 < (round(math.sqrt(((tag_corners[1][0] - tag_corners[0][0])**2) + ((tag_corners[1][1] - tag_corners[0][1])**2)))) < 200) and (100 < (round(math.sqrt(((tag_corners[2][0] - tag_corners[0][0])**2) + ((tag_corners[2][1] - tag_corners[0][1])**2)))) < 200):
       
        return tag_corners, True
    else:
        return tag_corners, False



def warping(img):

    tag_corners, cornersValid = findCorners(img)

    if cornersValid:

        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

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

        H_inv = np.divide((np.linalg.inv(H)), (np.linalg.inv(H))[2][2])

        warp_tag = np.empty((marker_height, marker_length))

        y_w = np.tile((np.arange(marker_length)), marker_height)
        x_w = np.repeat((np.arange(marker_height)), marker_length)

        w_cord = np.vstack((x_w, y_w, np.ones((y_w.shape))))

        cam_cord = np.matmul(H_inv, w_cord)

        cam_cord = (np.divide(cam_cord, cam_cord[-1])).astype(int)

        cam_cord_y = cam_cord[0]
        cam_cord_x = cam_cord[1]

        warp_tag[x_w, y_w] = thresh[cam_cord_x, cam_cord_y]

        warp_tag = cv2.flip(warp_tag, 0)

        return warp_tag, H_inv, marker_height, marker_length, True

    else:

        return None, None, None, None, False



def aruco_detect(img):

    warp_tag, _, __, ___, warp_valid = warping(img)

    if warp_valid:

        step_x, step_y = warp_tag.shape[0:2]

        step_x, step_y = step_x/8, step_y/8

        rotate_count = 0

        while (warp_tag[round(5.5*step_x + 5)][round(5.5*step_y + 5)] != 255 and rotate_count < 4):
            warp_tag = cv2.rotate(warp_tag, cv2.ROTATE_90_CLOCKWISE)
            rotate_count += 1

        aruco_binary = [warp_tag[round(3.5*step_x)][round(3.5*step_y)], warp_tag[round(3.5*step_x)][round(4.5*step_y)], warp_tag[round(4.5*step_x)][round(4.5*step_y)], warp_tag[round(4.5*step_x)][round(3.5*step_y)]]
        
        aruco_binary = np.sign(aruco_binary)

        aruco_id = 0
        for i in range(len(aruco_binary)):
            aruco_id += (2**i)*(aruco_binary[i])

        print("Aruco Marker ID is: ", aruco_id)

        return rotate_count, aruco_id, True

    else:
        return None, None, False



def aruco_align(frame, img):

    rotate_count, aruco_id, aruco_valid = aruco_detect(frame)

    if aruco_valid:

        count = 0

        while (count < rotate_count):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            count += 1

        return img, aruco_id, True

    else:
        return None, None, False


def testudo_image(frame, img):

    img, aruco_id, aruco_valid = aruco_align(frame, img)

    img = cv2.flip(img, 0)

    if aruco_valid:

        _, H_inv, marker_height, marker_length, __ = warping(frame)

        resized = cv2.resize(img, (marker_length, marker_height), interpolation = cv2.INTER_AREA)
        resized = cv2.medianBlur(resized, 3)

        y_w = np.tile((np.arange(marker_length)), marker_height)
        x_w = np.repeat((np.arange(marker_height)), marker_length)

        w_cord = np.vstack((x_w, y_w, np.ones((y_w.shape))))

        cam_cord = np.matmul(H_inv, w_cord)

        cam_cord = (np.divide(cam_cord, cam_cord[-1])).astype(int)

        cam_cord_y = cam_cord[0]
        cam_cord_x = cam_cord[1]

        frame[cam_cord_x, cam_cord_y] = resized[x_w, y_w]

        frame = cv2.medianBlur(frame,3)

        return frame, aruco_id, True

    else:
        return None, None, False


def track_image(video, img):

    img = cv2.imread(img)
    cap = cv2.VideoCapture(video)
    out = cv2.VideoWriter('part2a(2).avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))

    iter = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == False:
            break
        
        if iter != 657:

            final, id, valid = testudo_image(frame, img)

            if valid and id == 7:

                out.write(final)

                cv2.imshow('dst', final)
                cv2.waitKey(1)

        print(iter)
        iter += 1

    cap.release()
    out.release()

    cv2.destroyAllWindows()


