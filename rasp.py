import sys
import os
import shutil
import time
import datetime
import re
import csv
import cv2
import face_recognition
import numpy as np
import tkinter as tk

from types import MethodType
from functools import wraps
from collections import deque, defaultdict
from math import sqrt
from PIL import Image
from gpiozero import LED
from tkinter import ttk, N, S, E, W

class Profile:
    """Decorator able to time methods and classes,
    also able to count the number of calls
    """
    def __init__(self, func):
        wraps(func)(self)
        self.func = func
        self.ncalls = 0

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return MethodType(self, instance)

    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        start_time = time.time()
        print('Executing "{}" function'.format(self.func.__name__), end='... ')
        output = self.__wrapped__(*args, **kwargs)
        print('Sucess [{:.4f} seconds]'.format(time.time()-start_time))
        return output      

class Initializer:
    """ Initialize Raspberry Pi"""
    def __init__(self, gpio_pin, video_width, video_height):
        self.gpio_pin = gpio_pin
        self.video_width = video_width
        self.video_height = video_height
        self.imgpath = ('/home/pi/Desktop/tcc/image-for-loading/main_img.png')

    def __repr__(self):
        return (
            'Initializer(GPIO Pin: {0.gpio_pin!r}, '
            'Video Width: {0.video_width!r}, '
            'Video Height: {0.video_height!r})'.format(self)
        )

    def __str__(self):
        return (
            '(GPIO Pin: {0.gpio_pin!s}, '
            'Video Width: {0.video_width!s}, '
            'Video Height: {0.video_height!s})'.format(self)
        )

    def display_img(self):
        img = cv2.imread(self.imgpath)
        cv2.namedWindow('Reconhecimento Facial', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Reconhecimento Facial', 832, 624)
        cv2.imshow('Reconhecimento Facial', mat=img)
        cv2.waitKey(10)      

    def config_gpio(self):
        lock = LED(self.gpio_pin)
        lock.off()
        return lock

    def config_camera(self):
        camera = cv2.VideoCapture(index=0)
        camera.set(propId=3, value=self.video_width)
        camera.set(propId=4, value=self.video_height)
        flag_camera_on = True
        return camera, flag_camera_on

    @Profile
    def get_users_encodings(self):
        users_dir = '/home/pi/Desktop/tcc/reconhecimento-facial/users'
        users_encodings = {}
        for user in sorted(os.listdir(users_dir)):
            user_dir = os.path.join(users_dir, user)
            for imgfile in os.listdir(user_dir):
                imgpath = os.path.join(user_dir, imgfile)
                img = face_recognition.load_image_file(imgpath)
                img_height, img_width, img_depth = img.shape
                face_encodings = face_recognition.face_encodings(
                    face_image=img,
                    known_face_locations=[(0, img_width, img_height, 0)]
                )
                face_encodings = face_encodings[0]
                users_encodings.update({user: face_encodings})
        cv2.destroyAllWindows()
        return users_encodings

class Detector:
    """ Detector implemented through the Viola-Jones algorithm """
    def __init__(self):
        self.fpath = '/home/pi/Desktop/tcc/face detector/haarcascade_frontalface_default.xml'

    @Profile
    def config_face_detector(self):
        return cv2.CascadeClassifier(filename=self.fpath)

class Identifier:
    """ Real time facial identification """
    def __init__(self, lock, camera, flag_camera_on, users_encodings, face_detector):
        self.lock = lock
        self.camera = camera
        self.flag_camera_on = flag_camera_on
        self.users_encodings = users_encodings
        self.face_detector = face_detector        
        self.flag_enable_nn = True
        self.flag_enable_start_timer = False
        self.flag_same_user = False
        self.flag_enable_lock_off = False
        self.flag_smartphone = False
        self.chosen_user = None
        self.start_time_all = time.time()
        self.start_time = time.time()
        self.start_time_lock = time.time()
        self.start_time_change = time.time()
        self.initial_frame_time = 0
        self.min_distance = 0
        self.color = (255, 255, 255)
        self.nimgs = 0
        self.flag_rst = False
        self.flag_rst_noface = False
        self.flag_from_many_to_one = True
        self.flag_from_one_to_many = True
        self.flag_init_change = True
        self.last_x_pos = None
        self.flag_changed = False

    def get_frames(self, camera, face_detector):
        face_coord = ()
        flag_face_detected = False
        while not flag_face_detected:
            try:
                _, frame = camera.read()
                frame = cv2.flip(src=frame, flipCode=-1)
                gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
                face_coord = face_detector.detectMultiScale(
                    image=gray,
                    scaleFactor=1.2,
                    minNeighbors=5, 
                    minSize=(80, 80))
                flag_face_detected = True
            except cv2.error as error:
                print('cv2 error:', error)
            except:
                print('Unexpected error:', sys.exc_info()[0], sys.exc_info()[1])
        return frame, gray, face_coord

    def get_imgsize(self, frame, y_start, height, x_start, width, **kwargs):
        if len(kwargs) == 1:
            if 'cut' in kwargs:
                return frame[y_start+kwargs['cut']:(y_start+height)-kwargs['cut'],
                             x_start+kwargs['cut']:(x_start+width)-kwargs['cut']]
            elif 'expand' in kwargs:
                return frame[y_start-kwargs['expand']:(y_start+height)+kwargs['expand'],
                             y_start-kwargs['expand']:(y_start+width)+kwargs['expand']]
        else:
            return frame[y_start:y_start+height,
                         x_start:x_start+width]
       

    def rgb2gray(self, img):
        gray_value = 0.07 * img[:,:,2] + 0.72 * img[:,:,1] + 0.21 * img[:,:,0]
        gray_img = gray_value.astype(np.uint8)
        return gray_img  

    def analize_img_smartphone(self, captured_imgpath):
        img = face_recognition.load_image_file(captured_imgpath)
        if np.mean(img) >= 200:
            flag_smartphone = True
        else:
            flag_smartphone = False
        print('mean pixels =', np.mean(img))
        return flag_smartphone

    def adjust_gamma(self, imgpath, gamma):
        img = cv2.imread(imgpath)
        inv_gamma = 1 / gamma
        table = np.array([((i / 255) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype('uint8')
        new_img = cv2.LUT(img, table)
        new_imgpath = re.sub('.jpg', '', imgpath)
        new_imgpath = new_imgpath + '[gamma]' + '.jpg'
        cv2.imwrite(filename=new_imgpath, img=new_img)
        return new_imgpath

    def analize_img_gamma(self, captured_imgpath):
        img = face_recognition.load_image_file(captured_imgpath)
        if np.mean(img) <= 10:
            captured_imgpath = self.adjust_gamma(captured_imgpath, gamma=3)
        return captured_imgpath

    def capture_img(self, capture_img_params):
        (flag_enable_start_timer, flag_enable_nn, flag_smartphone, start_time, nimgs,
         frame, gray, x_start, y_start, width, height) = capture_img_params
        captured_imgpath = None
        if flag_enable_start_timer:
            start_time = time.time()
            flag_enable_start_timer = False
            flag_enable_nn = True
        if time.time() - start_time > 3 and flag_enable_nn == True:
            nimgs += 1
            captured_imgpath = ('/home/pi/Desktop/tcc/'
                                'reconhecimento-facial/temp/tmp{}.jpg'.format(nimgs))
            imgsize = self.get_imgsize(frame, y_start, height, x_start, width)
            cv2.imwrite(filename=captured_imgpath, img=imgsize)
            flag_smartphone = self.analize_img_smartphone(captured_imgpath)
            captured_imgpath = self.analize_img_gamma(captured_imgpath)
        return (
            start_time, flag_enable_start_timer, 
            flag_enable_nn, flag_smartphone, 
            nimgs, captured_imgpath
        )

    def save_user_data_one_textfile(self, min_distance, chosen_user):
        if min_distance >= 0.6:
            chosen_user = 'Indivíduo Desconhecido'
        d = datetime.datetime.now()
        fpath = ('/home/pi/Desktop/tcc/reconhecimento-facial/'
                 'users_history/textfiles/one_file/users_data.txt')
        with open(fpath, 'a') as f:
            f.write('\n{}, {}, {:.2f}'.format(
                d.strftime('%d/%m/%y, %H:%M'),
                chosen_user, 
                min_distance)
            )
        with open(fpath, 'r') as f:
            offset = 2
            lines = f.readlines()
            if len(lines) > 100 + offset:
                with open(fpath, 'w') as f:
                    f.write('Dia, Hora, Usuário, Distância\n\n')
                    for line in lines[1+offset:]:
                        f.write(line)

    def get_fileid(self, fname, first_delimiter='[', last_delimiter=']'):
        for index, char in enumerate(fname):
            if char == first_delimiter:
                first_index = index + 1
            elif char == last_delimiter:
                last_index = index - 1
        return ''.join(fname[index] for index in range(first_index, last_index+1))

    def get_last_fpath(self, fdir):
        fpaths = [os.path.join(fdir, fpath) for fpath in sorted(os.listdir(fdir))]
        return fpaths[len(fpaths)-1]

    def save_multiple_txtfiles(self, min_distance, chosen_user, d):
        fdir = '/home/pi/Desktop/tcc/reconhecimento-facial/users_history/textfiles/multiple_files'
        last_fpath = self.get_last_fpath(fdir)
        with open(last_fpath, 'r') as f:
            nlines = len(f.readlines())
        offset, max_num_users = 2, 1000
        if nlines < max_num_users + offset:
            with open(last_fpath, 'a') as f:
                f.write('\n{:03d}, {}, {}, {:.2f}'.format(
                    nlines-offset+1,
                    d.strftime('%d/%m/%y, %H:%M'),
                    chosen_user, min_distance))
        else:
            fileid = self.get_fileid(fname=last_fpath)
            last_fpath= re.sub(pattern=fileid, repl=str(int(fileid)+1), string=last_fpath, count=1)
            with open(last_fpath, 'w') as f:
                f.write('Id, Dia, Hora, Usuário, Distância\n\n')
                f.write('{:03d}, {}, {}, {:.2f}'.format(
                    nlines-offset+1,
                    d.strftime('%d/%m/%y, %H:%M'),
                    chosen_user, min_distance))

    def save_multiple_csvfiles(self, min_distance, chosen_user, d):
        fdir = '/home/pi/Desktop/tcc/reconhecimento-facial/users_history/csvfiles'
        last_fpath = self.get_last_fpath(fdir)
        with open(last_fpath, 'r', newline='') as csvfile:
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            nrows = len(rows)
        max_num_users = 1000
        if nrows < max_num_users:
            with open(last_fpath, 'a', newline='') as csvfile:
                writer = csv.DictWriter(
                    f=csvfile,
                    dialect='excel',
                    fieldnames=['Id', 'Dia', 'Hora',
                                'Usuário', 'Distância']
                    )
                writer.writerow({
                    'Id': '{:03d}'.format(nrows+1),
                    'Dia': d.strftime('%d/%m/%y'),
                    'Hora': d.strftime('%H:%M'),
                    'Usuário': chosen_user,
                    'Distância': '{:.2f}'.format(min_distance)
                })
        else:
            fileid = self.get_fileid(fname=last_fpath)
            last_fpath= re.sub(pattern=fileid, repl=str(int(fileid)+1), string=last_fpath, count=1)
            with open(last_fpath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(
                    f=csvfile,
                    dialect='excel',
                    fieldnames=['Id', 'Dia', 'Hora',
                                'Usuário', 'Distância']
                    )
                writer.writeheader()
                writer.writerow({
                    'Id': '{:03d}'.format(nrows+1),
                    'Dia': d.strftime('%d/%m/%y'),
                    'Hora': d.strftime('%H:%M'),
                    'Usuário': chosen_user,
                    'Distância': '{:.2f}'.format(min_distance)
                })
               

    def save_user_data_multiple_files(self, min_distance, chosen_user):
        if min_distance >= 0.6:
            chosen_user = 'Indivíduo Desconhecido'
        d = datetime.datetime.now()
        self.save_multiple_txtfiles(min_distance, chosen_user, d)
        self.save_multiple_csvfiles(min_distance, chosen_user, d)

    def parse_imgs(self, parse_imgs_params):
        (start_time, flag_enable_nn, min_distance, chosen_user, users_encodings,
         flag_rst_noface, captured_imgpath) = parse_imgs_params
        if time.time() - start_time > 3 and flag_enable_nn == True:
            flag_enable_nn = False
            nn_start_time = time.time()
            users_path = '/home/pi/Desktop/tcc/reconhecimento-facial/users'
            users_names = sorted(os.listdir(users_path))    
            try:
                img = face_recognition.load_image_file(captured_imgpath)
                img_height, img_width, img_depth = img.shape
                img_encodings = face_recognition.face_encodings(
                    face_image=img,
                    known_face_locations=[(0, img_width, img_height, 0)]
                )
                img_encodings = img_encodings[0]
                distances = {
                    key: face_recognition.face_distance([img_encodings], value)[0]
                    for key, value in users_encodings.items()
                }
                chosen_user = min(distances.keys(), key=(lambda k: distances[k]))
                min_distance = distances[chosen_user]
                flag_rst_noface = False
                self.save_user_data_multiple_files(min_distance, chosen_user)
            except IndexError as error:
                flag_rst_noface = True
                min_distance = 0
                print(error)
            except AttributeError as error:
                print(error)
            except Exception:
                for error in sys.exc_info():
                    print(error)
            os.unlink(captured_imgpath)
        time.sleep(0.001)
        return (
            start_time, flag_enable_nn, min_distance,
            chosen_user, flag_rst_noface
        )

    def draw_rectangle(self, frame, x_start, y_start, width, height, color):
        cv2.rectangle(img=frame, pt1=(x_start, y_start),
              pt2=(x_start+width, y_start+height),
              color=color,
              thickness=3)

    def write_name(self, frame, text, x_start, y_start, color):
        cv2.putText(img=frame, text=text, org=(x_start, y_start-10),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
            color=color, thickness=2)

    def write_msg_with_box(self, frame, text_x, text_y, text, color):
        if text == 'Carregando' or text == '':
            color = (0, 0, 0)
        elif text == 'Smartphone':
            color = (0, 0, 255)
        (text_width, text_height) = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2
        )
        (text_width, text_height) = (text_width, text_height)[0]
        box_coords = ((text_x, text_y), (text_x+text_width+20, text_y-text_height-20))
        cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        cv2.putText(
            frame, text, (text_x, text_y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=color, thickness=2
        )

    def write_distance(self, frame, min_distance, x_start, y_start, color):
        cv2.putText(
            img=frame, text=str('{:.5f}'.format(min_distance)),
            org=(x_start+200, y_start+100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=color)        

    def write_info_on_frame(
        self, min_distance, color, chosen_user, flag_smartphone,
        frame, x_start, y_start, width, height, only_one
        ):
        text = 'Carregando' if only_one else ''
        if min_distance == 0:
            color = (255, 255, 255)
        elif min_distance < 0.6 and not flag_smartphone:
            color = (255, 200, 0)
            text = chosen_user
        elif min_distance >= 0.6 and not flag_smartphone:
            text = 'Desconhecido'
            color = (0, 0, 255)
        elif flag_smartphone:
            color = (255, 255, 255)
            text = 'Smartphone'
        self.draw_rectangle(frame, x_start, y_start, width, height, color)
        self.write_msg_with_box(frame, x_start, y_start-10, text, color)
        #self.write_name(frame, text, x_start, y_start, color)
        #self.write_distance(frame, min_distance, x_start, y_start, color)
           
    def calculate_rec_distance(self, face_coords):
        video_width, video_height = 640, 480
        x1, y1 = video_width/2, video_height/2
        dists = {}
        for x_start, y_start, width, height in face_coords:
            x2, y2 = (x_start+width)/2, (y_start+height)/2
            dist = sqrt((x2-x1)**2 + (y2-y1)**2)
            dists.update({(x_start, y_start, width, height): dist})
        centered_coords = min(dists.keys(), key=lambda k: dists[k])
        del dists[centered_coords]
        cleared_coords = np.array(centered_coords)
        cleared_coords = cleared_coords.reshape((1, ) + cleared_coords.shape)
        rests = np.array([key for key in dists])
        return cleared_coords, rests

    def dealt_with_neck_bug(self, face_coords, left_center, right_center):
        count = 0
        flag_neck_bug = False
        for x_start, y_start, width, height in face_coords:
            if x_start >= left_center and x_start <= right_center:
                count += 1
        if count > 1:
            flag_neck_bug = True
        return flag_neck_bug

    def inform_change(self, face_coords, last_x_pos, dif, threshold_change):
        print('facecoods =', face_coords[0][0], ':::', 'lastxpos =', last_x_pos[0][0])
        print('dif=', dif)
        if dif >= threshold_change or dif <= -threshold_change:
            print('SHOULD RESET')

    def detect_change(self, face_coords, flag_init_change, last_x_pos, start_time, flag_rst):
        flag_rst = False
        if flag_init_change:
            flag_init_change = False
            last_x_pos = face_coords
        else:
            t = 0.25
            threshold_change = 10
            if time.time() - start_time > t:
                start_time = time.time()
                try:
                    for index in range(len(last_x_pos)):
                        dif = last_x_pos[index][0] - face_coords[index][0]
                        if dif >= threshold_change or dif <= -threshold_change:
                            flag_rst = True
                        #self.inform_change(face_coords, last_x_pos, dif, threshold_change)
                except IndexError as error:
                    pass
                except TypeError as error:
                    pass
                last_x_pos = face_coords
        return flag_init_change, last_x_pos, start_time, flag_rst

    def get_most_centered_face(self, face_coords, flag_enable_multiple_users):
        centered_coords = []
        rests_coords = ()
        left_center, right_center = 150, 300
        flag_neck_bug = self.dealt_with_neck_bug(face_coords, left_center, right_center)
        flag_init_change = False
        all_users_are_in_corner = True
        if flag_neck_bug:
            face_coords = ()
            flag_enable_multiple_users = True
            return face_coords, rests_coords, flag_enable_multiple_users
        for index, (x_start, y_start, width, height) in enumerate(face_coords):
            if x_start >= left_center and x_start <= right_center:
                centered_coords.append(face_coords[index])
                rests_coords = np.delete(arr=face_coords, obj=index, axis=0)
                all_users_are_in_corner = False
        if all_users_are_in_corner:
            centered_coords = None
            rests_coords = face_coords
            return centered_coords, rests_coords, flag_enable_multiple_users
        return np.array(centered_coords), rests_coords, flag_enable_multiple_users
       
    def inform_just_one_user(self, color, frame, face_coords):
        for x_start, y_start, width, height in face_coords:
            self.draw_rectangle(frame, x_start, y_start, width, height, color)
            self.write_msg_with_box(
                frame, text_x=150, text_y=50, 
                text='Apenas 1 Usuario', color=(0, 0, 0))        

    def open_lock(self, open_lock_params):
        (min_distance,  lock, flag_same_user, flag_enable_lock_off,
        start_time_lock) = open_lock_params
        t = 2
        if min_distance < 0.6 and min_distance > 0:
            if time.time() - start_time_lock <= t:
                lock.on()
            if not flag_same_user:
                flag_same_user = True
                flag_enable_lock_off = True
                start_time_lock = time.time()
        elif min_distance == 0:
            flag_same_user = False
            enable_lock_off = False
            lock.off()
        if flag_enable_lock_off:
            if time.time() - start_time_lock > t:
                lock.off()
        return lock, flag_same_user, flag_enable_lock_off, start_time_lock

    def close_window(self, camera):
        camera.release()
        cv2.destroyAllWindows()

    def config_gui(self, same_user_error=False, three_names_error=False):        
        def close(*args):
            if nametk.get() != '':
                root.destroy()                
        root = tk.Tk()
        root.title("Registro")
        root.minsize(300, 200)
        mainframe = ttk.Frame(root)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        if not same_user_error and not three_names_error:
            tk.Label(
                mainframe,
                text='Nome:',
                font=('Arial', 30)).grid(column=1, row=1, sticky=(N, W, E, S)
            )
            nametk = tk.StringVar()
            name_entry = ttk.Entry(
                mainframe,
                font=('Arial', 30),
                textvariable=nametk
            )
            name_entry.grid(column=2, row=1, sticky=(N, W, E, S))              
            button = tk.Button(
                mainframe,
                text='Registrar-se',
                font=('Arial', 30),
                command=close).grid(column=2, row=2, sticky=(N, W, E, S)
            )
        if same_user_error:
            tk.Label(
                mainframe,
                text='Nome já registrado. Tente novamente',
                bg='red',
                font=('Arial', 30)).grid(column=2, row=1, sticky=(N, W, E, S)
            )
            tk.Label(
                mainframe,
                text='Nome:',
                font=('Arial', 30)).grid(column=1, row=2, sticky=(N, W, E, S)
            )
            nametk = tk.StringVar()
            name_entry = ttk.Entry(
                mainframe,
                font=('Arial', 30),
                textvariable=nametk
            )
            name_entry.grid(column=2, row=2, sticky=(N, W, E, S))              
            button = tk.Button(
                mainframe,
                text='Registrar-se',
                font=('Arial', 30),
                command=close).grid(column=2, row=3, sticky=(N, W, E, S)
            )            
        if three_names_error:
            tk.Label(
                mainframe,
                text='Máximo três nomes. Tente novamente',
                bg='red',
                font=('Arial', 30)).grid(column=2, row=1, sticky=(N, W, E, S)
            )
            tk.Label(
                mainframe,
                text='Nome:',
                font=('Arial', 30)).grid(column=1, row=2, sticky=(N, W, E, S)
            )
            nametk = tk.StringVar()
            name_entry = ttk.Entry(
                mainframe,
                font=('Arial', 30),
                textvariable=nametk
            )
            name_entry.grid(column=2, row=2, sticky=(N, W, E, S))              
            button = tk.Button(
                mainframe,
                text='Registrar-se',
                font=('Arial', 30),
                command=close).grid(column=2, row=3, sticky=(N, W, E, S)
            )
        for child in mainframe.winfo_children():
            child.grid_configure(padx=10, pady=10)
        root.bind('<Return>', close)
        return root, nametk

    def convert_to_uppercase(self, name):
        spaces = 0
        for char in name:
            if char == ' ':
                spaces += 1
        if spaces == 2:
            first, middle, last = name.split(' ')
            if first[0].islower():
                first = re.sub(pattern=first[0], repl=first[0].upper(), string=first, count=1)
            if middle[0].islower():
                middle = re.sub(pattern=middle[0], repl=middle[0].upper(), string=middle, count=1)
            if last[0].islower():
                last = re.sub(pattern=last[0], repl=last[0].upper(), string=last, count=1)
            name = first + ' ' + middle + ' ' + last
        elif spaces == 1:
            first, last = name.split(' ')
            if first[0].islower:
                first = re.sub(pattern=first[0], repl=first[0].upper(), string=first, count=1)
            if last[0].islower:
                last = re.sub(pattern=last[0], repl=last[0].upper(), string=last, count=1)
            name = first + ' ' + last
        else:
            if name[0].islower:
                name = re.sub(pattern=name[0], repl=name[0].upper(), string=name, count=1)
        return name

    def insert_on_alphabetical_order(self, name, img, users_encodings):
        users = sorted(os.listdir('/home/pi/Desktop/tcc/reconhecimento-facial/users'))
        index = 0
        for user in users:
            for char_name, char_user in zip(name, user):
                if char_name > char_user:
                    index += 1
                    break
                elif char_name == char_user:
                    continue
                else:
                    break
        img_width, img_height, img_depth = img.shape
        new_user_enc = face_recognition.face_encodings(
            face_image=img,
            known_face_locations=[(0, img_width, img_height, 0)])
        new_user_enc = new_user_enc[0]        
        users_encodings.insert(index, new_user_enc)      
        return users_encodings

    def mk_user_registration(self, camera, face_detector, users_encodings):
        self.close_window(camera)    
        registration_photos_path = '/home/pi/Desktop/tcc/reconhecimento-facial/users'
        flag_registration_on = True
        flag_first_time = True
        flag_same_user = False
        flag_three_names = False
        while flag_registration_on:
            try:
                if flag_first_time:
                    flag_first_time = False
                    root, nametk = self.config_gui()
                elif flag_same_user:
                    flag_same_user = False
                    root, nametk = self.config_gui(same_user_error=True)
                elif flag_three_names:
                    flag_three_names = False
                    root, nametk = self.config_gui(three_names_error=True)
                root.mainloop()
                name = nametk.get()
                if name == '':
                    return users_encodings
                count = 0
                for char in name:
                    if char == ' ':
                        count += 1
                if count > 2:
                    raise ValueError
                name = self.convert_to_uppercase(name)
                users_names = os.listdir('/home/pi/Desktop/tcc/reconhecimento-facial/users')
                new_folder = os.path.join(registration_photos_path, name)
                os.mkdir(new_folder)
                flag_registration_on = False
            except FileExistsError as error:
                flag_same_user = True
            except ValueError as error:
                flag_three_names = True
        initializer = Initializer(gpio_pin=17, video_width=640, video_height=480)
        camera, flag_camera_on = initializer.config_camera()
        start_reg_time = time.time()
        flag_registration_on = True
        while flag_registration_on:
            frame, gray, face_coordenates = self.get_frames(camera, face_detector)
            cv2.imshow(winname='Registro', mat=frame)
            self.write_msg_with_box(
                frame, text_x=130, text_y=50,
                text=str('Olhe para a camera [{} s]'.format(int(10-(time.time()-start_reg_time)))),
                color=(0, 0, 0))
            for (x_start, y_start, width, height) in face_coordenates:
                self.draw_rectangle(frame, x_start, y_start, width, height, color=(255, 255, 255))
                if len(face_coordenates) > 1:
                    self.write_msg_with_box(
                        frame, text_x=330, 
                        text_y=450, text='Apenas 1 Usuario',
                        color=(0, 0, 0))
            cv2.imshow(winname='Registro', mat=frame)
            if time.time() - start_reg_time > 8:
                if isinstance(face_coordenates, np.ndarray) and len(face_coordenates) == 1:
                    imgpath = os.path.join(new_folder, name+'.jpg')
                    imgsize = self.get_imgsize(frame, y_start, height, x_start, width)
                    cv2.imwrite(filename=imgpath, img=imgsize)
                    img = face_recognition.load_image_file(imgpath)
                    try:
                        img_height, img_width, img_depth = img.shape
                        new_user_enc = face_recognition.face_encodings(
                            face_image=img,
                            known_face_locations=[(0, img_width, img_height, 0)])
                        new_user_enc = new_user_enc[0]        
                        users_encodings.update({name: new_user_enc})  
                        flag_registration_on = False
                    except IndexError as error:
                        flag_registration_on = True
                        start_reg_time = time.time()        
                else:
                    start_reg_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                shutil.rmtree(new_folder)
                break
        return users_encodings

    def exit_run(self, exit_run_params):
        camera, flag_camera_on, face_detector, users_encodings = exit_run_params
        flag_rst = False
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            flag_camera_on = False
        elif key == ord('r'):
            users_encodings = self.mk_user_registration(camera, face_detector, users_encodings)
            self.close_window(camera)
            initializer = Initializer(gpio_pin=17, video_width=640, video_height=480)
            camera, flag_camera_on = initializer.config_camera()            
            flag_rst = True
        return camera, flag_camera_on, users_encodings, flag_rst

    def reset(self):
        self.min_distance = 0
        self.start_time = time.time()
        self.flag_enable_start_timer = True
        self.flag_smartphone = False
        return (
            self.min_distance, 
            self.start_time,
            self.flag_enable_start_timer, 
            self.flag_smartphone
        )

    def inform(
        self, start_time_all, start_time, 
        start_time_lock, initial_frame_time, 
        disp_config=False, disp_frames=False
        ):
        if disp_config:
            print('program time = {:.2f}'.format(time.time() - start_time_all))
            print(10*'*', 'capture time = {:.2f}'.format(time.time() - start_time))
            print(20*'*', 'time lock =', time.time() - start_time_lock)
        if disp_frames:
            print("Each frame takes {:.4f} s. Result = {:.2f} fps".format(
                time.time() - initial_frame_time,
                1 / (time.time() - initial_frame_time))
            )
               
    def run(self):
        while self.flag_camera_on:
            frame, gray, face_coord = self.get_frames(self.camera, self.face_detector)
            if isinstance(face_coord, np.ndarray):
                if len(face_coord) == 1:
                    if self.flag_from_many_to_one:
                        self.flag_from_many_to_one = False
                        self.flag_from_one_to_many = True
                        self.flag_rst = True
                        (self.min_distance, self.start_time, self.flag_enable_start_timer,
                         self.flag_smartphone) = self.reset()
                    x_start, y_start, width, height = face_coord[0]
                    """ Capture mage """
                    capture_img_params =(
                        self.flag_enable_start_timer, self.flag_enable_nn, 
                        self.flag_smartphone, self.start_time, 
                        self.nimgs, frame, gray, 
                        x_start, y_start, width, height
                    )
                    (self.start_time, self.flag_enable_start_timer, 
                     self.flag_enable_nn, self.flag_smartphone,
                     self.nimgs, captured_imgpath) = self.capture_img(capture_img_params)
                    """ Parse users images """
                    parse_imgs_params = (self.start_time,self.flag_enable_nn, 
                                         self.min_distance, self.chosen_user,self.users_encodings,
                                         self.flag_rst_noface, captured_imgpath)
                    (self.start_time, self.flag_enable_nn, self.min_distance,
                     self.chosen_user, self.flag_rst_noface) = self.parse_imgs(parse_imgs_params)
                    """ Write appropriate information on the screen """
                    self.write_info_on_frame(
                        self.min_distance, self.color,
                        self.chosen_user, self.flag_smartphone,
                        frame, x_start, y_start, width, height, only_one=True
                    )        
                else:
                    if self.flag_from_one_to_many:
                        self.flag_from_one_to_many = False
                        self.flag_from_many_to_one = True
                        (self.min_distance, self.start_time, 
                         self.flag_enable_start_timer, self.flag_smartphone) = self.reset()
                    #self.inform_just_one_user(self.color, frame, face_coord)
                    (centered_coords, rests_coords, 
                     self.flag_rst) = self.get_most_centered_face(face_coord, self.flag_rst)
                    (self.flag_init_change, self.last_x_pos, 
                    self.start_time_change, self.flag_changed) = self.detect_change(
                        face_coord, self.flag_init_change,
                        self.last_x_pos, self.start_time_change, 
                        self.flag_changed)
                    if self.flag_changed:
                        (self.min_distance, self.start_time, self.flag_enable_start_timer,
                         self.flag_smartphone) = self.reset()
                    x_start, y_start, width, height = centered_coords[0]
                    """ Capture img """
                    capture_img_params =(
                        self.flag_enable_start_timer, self.flag_enable_nn,
                        self.flag_smartphone, self.start_time, self.nimgs, 
                        frame, gray, x_start, y_start, width, height
                    )
                    (self.start_time, self.flag_enable_start_timer, 
                     self.flag_enable_nn, self.flag_smartphone,
                     self.nimgs, captured_imgpath) = self.capture_img(capture_img_params)
                    """ Parse users images """
                    parse_imgs_params = (self.start_time,self.flag_enable_nn,self.min_distance,
                                         self.chosen_user,self.users_encodings, 
                                         self.flag_rst_noface, captured_imgpath)
                    (self.start_time, self.flag_enable_nn, self.min_distance,
                     self.chosen_user, self.flag_rst_noface) = self.parse_imgs(parse_imgs_params)
                    """ Write appropriate information on the screen """
                    self.write_info_on_frame(
                        self.min_distance, self.color,
                        self.chosen_user, self.flag_smartphone,
                        frame, x_start, y_start, width, height, only_one=True
                    )
                    """ Render a black color for users away from the center """
                    for (x_start, y_start, width, height) in rests_coords:
                        self.draw_rectangle(frame, x_start, y_start, width, height, color=(0, 0, 0))
            elif isinstance(face_coord, tuple):
                self.flag_enable_multiple_users = True
                (self.min_distance, self.start_time,
                 self.flag_enable_start_timer, self.flag_smartphone) = self.reset()
            else:
                raise Exception("An error ocurred with facial detection," 
                                "'face_coords' should return tuple or ndarray")
            open_lock_params = (
                self.min_distance,  self.lock,
                self.flag_same_user, self.flag_enable_lock_off, 
                self.start_time_lock
            )
            (self.lock, self.flag_same_user, self.flag_enable_lock_off,
             self.start_time_lock) = self.open_lock(open_lock_params)
            cv2.imshow(winname='Reconhecimento Facial', mat=frame)
            exit_run_params = (
                self.camera, self.flag_camera_on, 
                self.face_detector, self.users_encodings
            )
            (self.camera, self.flag_camera_on, 
             self.users_encodings, self.flag_rst) = self.exit_run(exit_run_params)
            if self.flag_rst or self.flag_rst_noface:
                self.flag_rst = False
                self.flag_rst_noface = False
                (self.min_distance, self.start_time, 
                 self.flag_enable_start_timer, self.flag_smartphone) = self.reset()
            self.inform(
                self.start_time_all, self.start_time, 
                self.start_time_lock, self.initial_frame_time
            )
            self.initial_frame_time = time.time()
        self.close_window(self.camera)

def main():
    initializer = Initializer(gpio_pin=17, video_width=640, video_height=480)
    initializer.display_img()
    lock = initializer.config_gpio()
    camera, flag_camera_on = initializer.config_camera()
    users_encodings = initializer.get_users_encodings()
    detector = Detector()
    face_detector = detector.config_face_detector()
    identifier = Identifier(lock, camera, flag_camera_on, users_encodings, face_detector)
    identifier.run()  

if __name__ == '__main__':
    main()
	
	
