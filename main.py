import torch
import cv2
from PIL import ImageGrab, Image
import numpy as np
import pyautogui as pg
from time import sleep
from typing import Tuple

class Brain():
    decoder = [str(i) for i in range(10)] + ['+', '-', '*', '/']
    health = 100

    def __init__(self, path:str, task_box:Tuple[int, int, int, int],
                number_height:int, restart_button:Tuple[int, int],
                left_button:Tuple[int, int], right_button:Tuple[int, int],
                healthbar:Tuple[int, int, int]) -> None:
        """Brain Class init

        Parameters:
        path (str): classification model path ("model/v3" is best)
        task_box: grab screen (x,y,x,y)
        number_height: letter height (needed for correct resize)
        restart_button: positin of restart button, this pixel should be green
        left_button: left button coordinates
        right_button: right button coordinates
        healthbar: progressbar (x, y, length)

        """
        self.model = torch.load(path)
        self.model = self.model.cpu()
        self.task_box = task_box
        self.number_height = number_height
        self.restart_button = restart_button
        self.left_button = left_button
        self.right_button = right_button
        self.healthbar = healthbar
        

    def predict(self, img:np.array) -> str:
        #print(img.shape)
        #img = self.prepare_image(img)
        self.model.eval()
        torch.no_grad()
        res = []
        for i in self.model.forward(img):

            preds = list(i)
            ind = preds.index(max(preds))
            res.append(self.decoder[ind])
        return res

    def prepare_image(self, img):
        img = img / 255
        img = np.expand_dims(img, 2)
        img = np.expand_dims(img, 0)
        #print(img.shape, 'prep')
        img = np.rollaxis(img, 3, 1)
        #print(img.shape, 'prep')
        
        img = torch.tensor(img, dtype=torch.float32)
        
        return img

    def edit_image(self, img: np.array) -> np.array:
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #print(img.shape)
        h, w = img.shape
        img = cv2.resize(img, (int(w * (30 / self.number_height)), int(h * (30 / self.number_height))))
        h, w = img.shape
        dw = int((30 - w) / 2)
        dh = int((30 - h) / 2)
        dw2 = 30 - w - dw
        dh2 = 30 - h - dh
        img = cv2.copyMakeBorder(img, dh, dh2 , dw, dw2, cv2.BORDER_CONSTANT, 0)
        return img
    
    def _dist(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
    def get_health(self):
        screen = np.array(ImageGrab.grab(bbox=(self.healthbar[0], self.healthbar[1],
                                            self.healthbar[0] + self.healthbar[2], self.healthbar[1] + 2)))
        thresh = np.array(cv2.inRange(screen, (200, 200, 200), (255, 255, 255)))
        output = cv2.connectedComponentsWithStats(thresh, 2, cv2.CV_32S)[2]
        health = output[-1][2]
        #print(health / self.healthbar[2])
        return health / self.healthbar[2]

    def save_log(self, img, imgname, text):
        img = cv2.putText(img, text, (0, self.task_box[3] - self.task_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imwrite(f'logs/{imgname}.png', img)

    def think(self, src='screen', chck=False, loop_id=0):
        """Think funuction
        solve quuix and click answer.

        Parameters:
        src (str): screen to get image from screen or path to image
        chck (bool): check restart button, and click it
        loop_id (bool): required to save image in log folder

        """
        if src != 'screen':
            img = cv2.imread(src)
        else:
            img = self.capture_screen(chck=chck)

        thresh = np.array(cv2.inRange(img, (250, 250, 250), (255, 255, 255)))
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        stat = output[2]
        imagesc = []
        #y_cords = [] used for another method of sorting
        for element in stat[1:]:
            start_point = (element[0], element[1])
            end_point = (element[0] + element[2], element[1] + element[3])
            #y_cords.append(end_point[1])
            imagesc.append(start_point + end_point)
        #y_deltas = []
        #for i in range(len(y_cords)-1):
        #    y_deltas.append(y_cords[i + 1] - y_cords[i])
        #mx = y_deltas.index(max(y_deltas))
        mid_y = (self.task_box[3] - self.task_box[1]) / 2
        #(y_deltas[mx] + y_deltas[mx + 1]) * 0.5
        imagesc.sort(key=lambda x:(x[1] > mid_y) * 9999 + x[0])
        last_symb = 0
        
        for i, (x, y, xx, yy) in enumerate(imagesc): 
            if i == 0:
                images = self.prepare_image(self.edit_image(thresh[y:yy, x:xx]))
                t2 = cv2.rectangle(thresh.copy(), (x, y), (xx, yy), (255, 255, 255), thickness=3)
            else:
                images = torch.vstack((images, self.prepare_image(self.edit_image(thresh[y:yy, x:xx]))))
                t2 = cv2.rectangle(t2, (x, y), (xx, yy), (255, 255, 255), thickness=3)

        preds = self.predict(images)
        
        v = ''.join(preds).replace('--', '==').replace('**', '==')
        #print(v)
        e = True
        try:
            e = eval(v)
        except Exception as err:
            print(err)
            self.save_log(thresh, loop_id, v + str(err))
        if e:
            self.click(0)
        else:
            self.click(1)
        #sleep(0.1)
        h = self.get_health()
        if 0.3 < self.health - h:
            # if answer is incorrect
            print(v)
            self.save_log(thresh, loop_id, v)
        self.health = h

    def capture_screen(self, chck=False):
        if chck:
            screen = np.array(ImageGrab.grab(bbox=(self.restart_button[0], self.restart_button[1],
                                            self.restart_button[0]+ 2, self.restart_button[1] + 2,
            )))
            if screen[1][1][0] < 140:
                pg.click(*self.restart_button)
                sleep(0.1)
                print('restart')
        return np.array(ImageGrab.grab(bbox=self.task_box))
        


    def click(self, a):
        if a == 0:
            pg.click(*self.left_button)
        else:
            pg.click(*self.right_button)

    def imshow(self, img):
        #return
        cv2.imshow('img', img)
        while True:
            
            k = cv2.waitKey(1)
            if k == 27 or False:
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    brain = Brain('model/v3',
                    (740, 180, 1420, 460),  # grab screen (x,y,x,y)
                    90,  # letter height (needed for correct resize)
                    (1070, 730),  # positin of restart button, this pixel should be green
                    (930, 740),  # left button coordinates
                    (1220, 740),  # right button coordinates
                    (750, 510, 650)  # progressbar (x, y, length)
                )
    sleep(10)
    cnt = 0
    while True:
        try:
            brain.think(chck=cnt % 20 == 0, loop_id=cnt)
        except SyntaxError as e:
            print(e.with_traceback)
        cnt += 1


__all__ = [Brain]
__version__ = '0.1'
