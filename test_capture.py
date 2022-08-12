import cv2
print("Loading the Camera......")
print(" ")
print("If you want to exit,please press (ESC) !!")
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 5.0, (1024, 1024))

while (cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape) # (720*1280*3)
    flag = cv2.waitKey(1)
    
    if flag == 27:
        print("Quit !!")

        cap.release()
        cv2.destroyAllWindows()
        break