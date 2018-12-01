from skimage import io      # Importing the io module from the scikit-image library.
import threading
import cv2

running = True
frame = 0
cam = cv2.VideoCapture(0)
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
    """The make_720p function sets the resolution of the screen to 720p."""
    cam.set(3, 1280)
    cam.set(4, 720)


def access_camera():
    global running
    global frame
    while True:
        running, frame = cam.read()
        

def save_and_show():
    while True:
        if running:
            cv2.imwrite('image/now.jpg', frame)
            img = cv2.imread('image/now.jpg', 1)

            if verify_image('image/now.jpg'):
                # Creating a copy of the imgage in grey scale.
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Creating a tuple with a list of detected faces within the image.
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                # This for loop iterates thought the tuple to locate the detected faces within the image.
                for (x, y, w, h) in faces:
                    # Detecting one face at the time by setting each detected face equals to the first face coordinates.
                    (x, y, w, h) = faces[0]
                    # Printing the coordinates given by the face.
                    print(x, y, w, h)

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

    cam.release()
    cv2.destroyAllWindows()  # Destroying all created windows.


def show_():
    while True:
        img = cv2.imread('image/now.jpg', 1)
        cv2.imshow('Frame', img)
        cv2.waitKey(1)


def main():
    if __name__ == "__main__":
        t1 = threading.Thread(target=access_camera)
        t2 = threading.Thread(target=save_and_show)
        # t3 = threading.Thread(target=show_)

        t1.start()
        t2.start()
        # t3.start()

        t1.join()
        t2.join()
        # t3.join()


main()
