import cv2


def show_colorpic(color_pic, epoch):
    cv2.imshow(f"epoch{epoch}_result", color_pic)
    cv2.waitKey(2)
