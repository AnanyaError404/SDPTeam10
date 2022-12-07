import cv2
import time
import math as m
import mediapipe as mp
import sys
from threading import Thread
from multiprocessing import process
from subprocess import Popen, PIPE, CalledProcessError
import pyttsx3

flag = "hiii"

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""

def sendWarning():
    #print("Posture is bad!!!")
    pass

# =============================CONSTANTS and INITIALIZATIONS=====================================#

# ===============================================================================================#

class Poses:
    def __init__(self):#self, file_name, cap, fps, width, heigth, frame_size, fourcc, height, video_output):
        self.file_name = 0
        self.cap = cv2.VideoCapture(0)

        self.test = "hiii"

        # Meta.
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (self.width, self.height)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Video writer.
        self.video_output = cv2.VideoWriter('output.mp4', self.fourcc, self.fps, self.frame_size)

        # Initilize frame counters.
        self.good_frames = 0
        self.mid_frames = 0
        self.bad_frames = 0

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Colors.
        self.blue = (255, 127, 0)
        self.red = (50, 50, 255)
        self.green = (127, 255, 0)
        self.dark_blue = (127, 20, 0)
        self.light_green = (127, 233, 100)
        self.yellow = (0, 255, 255)
        self.pink = (255, 0, 255)

        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def Squat(self):

        while self.cap.isOpened():
            # Capture frames.
            success, image = self.cap.read()
            if not success:
                print("Null.Frames")
                break
            # Get fps.
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Get height and width.
            h, w = image.shape[:2]

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image.
            keypoints = self.pose.process(image)

            # Convert the image back to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Use lm and lmPose as representative of the following methods.
            lm = keypoints.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.      
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Left knee
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), self.font, 0.9, self.green, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), self.font, 0.9, self.red, 2)

            # Calculate angles by defining the nodes to take angles of.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            knee_inclination = findAngle(l_hip_x, l_hip_y, l_knee_x, l_knee_y)
            #leg_pos = finda

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, self.dark_blue, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, self.dark_blue, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, self.dark_blue, -1)
            #cv2.circle(image, (r_shldr_x, r_shldr_y), 7, self.pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, self.dark_blue, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, self.dark_blue, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination < 45 and 10 < torso_inclination < 60:
                self.bad_frames = 0
                self.mid_frames = 0
                self.good_frames += 1
                
                cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(knee_inclination)), (l_knee_x + 10, l_knee_y), self.font, 0.9, self.light_green, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), self.green, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), self.green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), self.green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), self.green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), self.green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y - 100), self.green, 4)
                

            elif 45 <= neck_inclination < 60  or 60 <= torso_inclination < 65:
                self.good_frames = 0
                self.mid_frames += 1
                self.bad_frames = 0

                cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, self.yellow, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), self.font, 0.9, self.yellow, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.yellow, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.yellow, 2)
                cv2.putText(image, str(int(knee_inclination)), (l_knee_x + 10, l_knee_y), self.font, 0.9, self.yellow, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), self.yellow, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), self.yellow, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), self.yellow, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), self.yellow, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), self.yellow, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y - 100), self.yellow, 4)
                
                
            
            else:
                self.good_frames = 0
                self.mid_frames = 0
                self.bad_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(knee_inclination)), (l_knee_x + 10, l_knee_y), self.font, 0.9, self.red, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), self.red, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), self.red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), self.red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), self.red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), self.red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y - 100), self.red, 4)

            # Calculate the time of remaining in a particular posture.
            global good_time, bad_time, mid_time
            good_time = (1 / fps) * self.good_frames
            bad_time =  (1 / fps) * self.bad_frames
            mid_time =  (1 / fps) * self.mid_frames

            # Pose time.
            if good_time > 0:
                exp = True
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), self.font, 0.9, self.green, 2)
                #global flag
                flag == "Good"
                print(good_time)

            elif mid_time > 0:
                exp = True
                time_string_good = 'Deviated Posture Time : ' + str(round(mid_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), self.font, 0.9, self.yellow, 2)

            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), self.font, 0.9, self.red, 2)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            if bad_time > 3:
                exp = False
                sendWarning()
            # Write frames.
            self.video_output.write(image)

            # Display.
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            pass

        self.cap.release()
        cv2.destroyAllWindows()

    def T_Pose(self):
        time.sleep(1)
        while self.cap.isOpened():
            # Capture frames.
            success, image = self.cap.read()
            if not success:
                print("Null.Frames")
                break
            # Get fps.
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Get height and width.
            h, w = image.shape[:2]

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image.
            keypoints = self.pose.process(image)

            # Convert the image back to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Use lm and lmPose as representative of the following methods.
            lm = keypoints.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.      
            
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            
            # Wrists.
            l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
            
            r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
            r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

            #Elbows
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            
            r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            #if offset < 100:
            #    cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), self.font, 0.9, self.green, 2)
            #else:
            #    cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), self.font, 0.9, self.red, 2)

            # Calculate angles by defining the nodes to take angles of.
            leftelbow_inclination = findAngle(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)
            rightelbow_inclination = findAngle(r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)
            Lwrist_elbow_inclination = findAngle(l_wrist_x, l_wrist_y, l_elbow_x, l_elbow_y)
            Rwrist_elbow_inclination = findAngle(r_wrist_x, r_wrist_y, r_elbow_x, r_elbow_y)
            #torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            #leg_pos = finda

            # Draw landmarks.
            #cv2.circle(image, (l_shldr_x, l_shldr_y), 7, self.yellow, -1)
            #cv2.circle(image, (l_ear_x, l_ear_y), 7, self.yellow, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, self.blue, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, self.blue, -1)
            cv2.circle(image, (l_elbow_x, l_elbow_y), 7, self.blue, -1)
            cv2.circle(image, (r_elbow_x, r_elbow_y), 7, self.blue, -1)
            cv2.circle(image, (l_wrist_x, l_wrist_x), 7, self.blue, -1)
            cv2.circle(image, (r_wrist_x, r_wrist_y), 7, self.blue, -1)

            
            #cv2.circle(image, (l_hip_x, l_hip_y), 7, self.yellow, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            #cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, self.yellow, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'elbow : ' + str(int(leftelbow_inclination))# + '  Torso : ' + str(int(torso_inclination))
            angle_text_string = 'elbow : ' + str(int(rightelbow_inclination))
            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            

            if 90 < Lwrist_elbow_inclination <= 105 and 90 < Rwrist_elbow_inclination <= 105 :
                self.bad_frames = 0
                self.good_frames += 1


                cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(leftelbow_inclination)), (l_shldr_x + 10, l_shldr_y), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(rightelbow_inclination)), (r_shldr_x + 10, r_shldr_y), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(Lwrist_elbow_inclination)), (l_elbow_x + 10, l_elbow_y), self.font, 0.9, self.light_green, 2)
                cv2.putText(image, str(int(Rwrist_elbow_inclination)), (r_wrist_x, r_elbow_y), self.font, 0.9, self.light_green, 2)
                #cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.light_green, 2)

                # Join landmarks.
                cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y), self.green, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), self.green, 4)
                cv2.line(image, (r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), self.green, 4)
                cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), self.green, 4)
                cv2.line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), self.green, 4)

            else:
                self.good_frames = 0
                self.bad_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(leftelbow_inclination)), (l_shldr_x + 10, l_shldr_y), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(rightelbow_inclination)), (r_shldr_x + 10, r_shldr_y), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(Lwrist_elbow_inclination)), (l_elbow_x + 10, l_elbow_y), self.font, 0.9, self.red, 2)
                cv2.putText(image, str(int(Rwrist_elbow_inclination)), (r_wrist_x, r_elbow_y), self.font, 0.9, self.red, 2)
                #cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), self.font, 0.9, self.light_green, 2)

                # Join landmarks.
                cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y), self.red, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_elbow_x, l_elbow_y), self.red, 4)
                cv2.line(image, (r_shldr_x, r_shldr_y), (r_elbow_x, r_elbow_y), self.red, 4)
                cv2.line(image, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), self.red, 4)
                cv2.line(image, (r_wrist_x, r_wrist_y), (r_elbow_x, r_elbow_y), self.red, 4)


            # Write frames.
            self.video_output.write(image)

            # Display.
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            pass

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    p = Poses()
    
    selection = input ("Select pose: ")
    
    if selection == "Tpose":
        engine = pyttsx3.init()
        engine.say("For a T pose, simply straighten your arms such that they are perpendicular with your body.")
        engine.runAndWait()
        p.T_Pose()

        
    elif selection == "Squat" or "squat":
        engine = pyttsx3.init()
        engine.say("Please face the camera with the left side of your body.   For a squat pose, stand with your feet slightly wider than your hips. Look straight ahead and pick a spot on the wall in front of you. Then drive your knees down to the floor while keeping your spine in a neutral position throughout the motion.")
        engine.runAndWait()
        p.Squat()

    
        
    '''
    proc=subprocess.Popen('echo "helloooo"', shell=True, stdout=subprocess.PIPE, )
    output=proc.communicate()[0]
    print(output)
    '''

        
    
    
    # For webcam input replace file name with 0.
    

    






