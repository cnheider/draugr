import cv2

from draugr.opencv_utilities.namespaces.enums import CameraPropertyEnum

cameraNumber = 1
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
cap = cv2.VideoCapture()
cap.open(cameraNumber + cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FOURCC, fourcc)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 3840
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 2160
# cap.set(cv2.CAP_PROP_FPS, 60) # 30
# cap.set(CameraPropertyEnum.FOURCC.value, cv2.VideoWriter.fourcc('Y','1','6',' '))
cap.set(CameraPropertyEnum.convert_rgb.value, 0)
cap.set(CameraPropertyEnum.auto_focus.value, 0)
cap.set(CameraPropertyEnum.settings.value, 1)
# cap.set(CameraPropertyEnum.mode.value, 0)

from draugr.opencv_utilities import frame_generator, to_gray
from draugr.opencv_utilities.windows.image import show_image
from draugr.tqdm_utilities import progress_bar

if __name__ == "__main__":
    for image in progress_bar(frame_generator(cap)):
        # gray = to_gray(image)
        # show_image(gray)
        if show_image(image, wait=1):
            break  # esc to quit
