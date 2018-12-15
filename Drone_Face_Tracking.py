from skimage import io      # Importing the io module from the scikit-image library.
import threading            # Importing the threading module to multi-thread the code.
import cv2                  # Importing opencv-python to work on video face recognition.


class Frame:
    pass


# Initializing the global variables that will be used throughout the script.
# running = True                  # running global set to true.
# frame = 0                       # frame global set to 0.
# Creating a VideoCapture object with the default camera passed as its argument.
cam = cv2.VideoCapture(0)       # 'tcp://192.168.1.1:5555'
fr = Frame
fr.running = True
fr.frame = 0
# Creating a CascadeClassifier object to identify frontal faces.
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


def verify_image(img_file):
    """
    The verify_image function verifies whether the given image is complete or not.
    :param img_file: the image to verify.
    :return: True if the image is complete or False otherwise.
    """
    try:
        image = io.imread(img_file)
    except:
        return False
    return True


def make_720p():
    """
    The make_720p function sets the resolution of the screen to 720p.
    """
    cam.set(3, 1280)
    cam.set(4, 720)


def access_camera():
    """
    The access_camera function accesses the camera to get the given frame.
    """

    # global running      # Declaring the running global variable to be able to change it.
    # global frame        # Declaring the frame global variable to be able to change it.

    # This while loop gets a frame per iteration of the loop.
    while True:
        running, frame = cam.read()
        fr.running = running
        fr.frame = frame


def save_and_show():
    """
    The save_and_show function saves the access_camera function's given frame, shows it and
    recognizes the faces within the frame.
    """
    # This while loop saves the given frame, takes it back to show it up in the screen and executes the
    # face recognition operations on the given frame.
    while True:
        running = fr.running
        frame = fr.frame
        if running:     # If the frame was successfully read.
            cv2.imwrite('image/now.jpg', frame)         # Saving frame into the system as an image.
            img = cv2.imread('image/now.jpg', 1)        # Reading the image back from the system.

            if verify_image('image/now.jpg'):       # If the image is not corrupted.
                # Creating a copy of the imgage in grey scale.
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Creating a tuple with a list of detected faces within the image.
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                # This for loop iterates thought the tuple to locate the detected faces within the image.
                for (x, y, w, h) in faces:
                    # Detecting one face at the time by setting each detected face equals to the first face coordinates.
                    (x, y, w, h) = faces[0]
                    # Printing the coordinates given by the face.
                    # print(x, y, w, h)

                    color = (255, 0, 0)  # To hold the color of the rectangle that will be drawn around the face.
                    stroke = 2  # To hold the stroke of the rectangle.
                    end_cord_x = x + w  # To hold the ending coordinate on the x axis.
                    end_cord_y = y + h  # To hold the ending coordinate on the y axis.
                    # Crating rectangle to specify face's position.
                    cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)
                # Showing the frame up on the screen.
                cv2.imshow('frame', img)
                # This if statements breaks the loop if the "q" key is pressed.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cam.release()            # Releasing VideoCapture object.
    cv2.destroyAllWindows()  # Destroying all created windows.


def main():
    if __name__ == "__main__":      # If the current executing thread is the main function.
        # Creating the first thread with the access_camera function as its target.
        t1 = threading.Thread(target=access_camera)
        # Creating the second thread with the save_and_show function as its target.
        t2 = threading.Thread(target=save_and_show)

        t1.start()      # Running the first thread.
        t2.start()      # Running the second thread.

        t1.join()       # Waiting until the first thread finishes before continuing.
        t2.join()       # Waiting until the second thread finishes before continuing.


main()

