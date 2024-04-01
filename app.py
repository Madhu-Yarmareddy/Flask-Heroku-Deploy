from flask import Flask, request, jsonify
import numpy as np
from mediapipe import solutions
import cv2

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_input():
    path = request.data.decode('utf-8')
    nfsize=25
    sz=150
    mp_drawing=solutions.drawing_utils
    mpDraw=solutions.drawing_utils
    mp_pose=solutions.pose
    mpPose=solutions.pose
    pose = mpPose.Pose()
    mpHands = solutions.hands
    hands = mpHands.Hands()
    x=[]
    size=nfsize
    count=-1
    # iterating through the video
    cap = cv2.VideoCapture(path) # opening the video
    if cap.isOpened():  # checking if the video is opened
        # print("Video is opened")
        nfs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//size # calculating the number of frames to be skipped
        if nfs==0: # if the number of frames to be skipped is 0
            nfs=1 # setting the number of frames to be skipped to 1
        x1,x2,x3=[],[],[] # creating the empty lists
        fps = (cap.get(cv2.CAP_PROP_FPS)) # getting the fps of the video
        while True: # iterating through the frames
            ret,frame = cap.read()
            if not ret: # if the frame is not read
                break # break the loop
            # resizing the frame
            frame=cv2.resize(frame,(600,600))
            img1=np.zeros(frame.shape)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for handlms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img1, handlms, mpHands.HAND_CONNECTIONS)
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Render detections
            mp_drawing.draw_landmarks(img1, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            # storing img1 in frame variable
            frame=img1
            # getting the frame count
            frmid=cap.get(1)
            # resizing the frame
            frame=cv2.resize(frame,(sz,sz))
            # appending the frame to the list
            if frmid%nfs==0 and ret:
                x1.append(frame/255)
                # x2.append(np.array(cv2.flip(frame,1))/255) # flipping the frame

    # appending the frames to the list
    if len(x1)!=size:
        if len(x1)<size:
            x1.append(x1[-(size-len(x1))])
            # x2.append(x2[-(size-len(x1))])
        else:
            x1=x1[0:size]
            # x2=x2[0:size]
    x.append(x1)
    # x.append(x2)
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # np.expand_dims(np., axis=0)
    return jsonify(output=np.array(x))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
