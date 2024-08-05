import cv2
import numpy as np
import os
import sys
import torch
from concurrent.futures import ThreadPoolExecutor
from camera_calibration import calib, undistort
from image_processing import get_combined_gradients, get_combined_hls, combine_grad_hls
from detect_line import Line, get_perspective_transform, get_lane_lines_img, illustrate_driving_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from parameter import ProcessingParams
import pandas as pd
import time
from codecarbon import EmissionsTracker,OfflineEmissionsTracker

# Variabili globali
frames_to_save = []  # Lista per memorizzare i frame da salvare
MAX_FRAMES_TO_SAVE = 50 #Immagini da salvare
#frames_to_save = deque(maxlen=MAX_FRAMES_TO_SAVE) # Coda per memorizzare i frame da salvare
detected_front_vehicles = []
detected_vehicles = {}
vehicle_counter = 0
current_offsets={}
target_offsets = {}
destra = False
sinistra = False
use_car_fix = False
performance_data = pd.DataFrame(columns=['Frame', 'Processing Time'])


# This method load yolo_model with PyTorch Hub
def load_yolo_model(yolo_repo_path, model_path):
    sys.path.append(yolo_repo_path)
    yolo_model = torch.hub.load(yolo_repo_path, 'custom', path=model_path, source='local')
    return yolo_model

# This method detects objects in a frame using a YOLO model.
def detect_objects(frame, yolo_model):
    yolo_outputs = yolo_model(frame)
    output = yolo_outputs.xyxy[0]
    return output

def gradual_mirror(img, factor):
    height, width = img.shape[:2]
    
    # Creazione della maschera di gradiente per il controllo dello specchiamento
    mask = np.ones_like(img, dtype=np.float32)
    
    # Calcolo della dimensione dell'area di specchiamento
    mirror_width = int(width * factor)
    
    # Applicazione della maschera per ottenere lo specchiamento graduale
    if factor > 0 and factor < 1:
        mask[:, :mirror_width] = np.linspace(0, 1, mirror_width)[np.newaxis, :, np.newaxis]
    elif factor >= 1 and factor < 2:
        mask[:, :mirror_width] = np.ones((height, mirror_width, 3), np.float32)
    else:
        mask[:, :mirror_width] = np.linspace(1, 0, mirror_width)[np.newaxis, :, np.newaxis]

    gradual_mirror_img = img * mask
    
    return gradual_mirror_img.astype(np.uint8)

def overlay_png(image, coordinates, labels, window_scale_factor, car_back_img, car_back_imgS, car_front_imgS, car_front_img, stop_img, confidence, moto_back, moto_backS, distance_m, car_back_imgM, car_front_imgM, moto_back_imgM,truck_back_img,truck_back_imgS,truck_back_imgM):
    """
     Questa funzione prende un'immagine e una serie di parametri e esegue varie operazioni di elaborazione delle immagini per rilevare e visualizzare veicoli e linee di corsia. Restituisce l'immagine processata.
    
    Parametri:
    - image: l'immagine su cui applicare le sovrapposizioni
    - coordinates: le coordinate del rettangolo che delimita l'oggetto da sovrapporre
    - labels: l'etichetta dell'oggetto da sovrapporre
    - window_scale_factor: il fattore di scala della finestra
    - car_back_img, car_back_imgS, car_front_imgS, car_front_img, stop_img, moto_back, moto_backS, distance_m, car_back_imgM, car_front_imgM, moto_back_imgM, truck_back_img, truck_back_imgS, truck_back_imgM: le immagini dei diversi oggetti da sovrapporre
    
    Restituisce:
    - L'immagine con le sovrapposizioni dei veicoli e delle linee di corsia
    
    Descrizione:
    - La funzione inizia definendo un dizionario che associa le etichette delle immagini ai rispettivi file PNG.
    - Viene effettuato un controllo sulla confidenza del rilevamento e sull'esistenza dell'etichetta nel dizionario. Se uno dei due controlli fallisce, la funzione termina.
    - Viene selezionata l'immagine PNG corrispondente all'etichetta.
    - Vengono calcolate le dimensioni dell'immagine di sfondo e le coordinate del rettangolo.
    - Viene calcolato l'offset verticale in base alla distanza dell'oggetto.
    - Vengono effettuate diverse operazioni di ridimensionamento e rotazione dell'immagine PNG in base alla distanza e all'etichetta dell'oggetto.
    - Viene effettuato un controllo per evitare che l'area di sovrapposizione ecceda i limiti dell'immagine di sfondo.
    - Viene applicata la sovrapposizione dell'immagine PNG sull'immagine di sfondo utilizzando una maschera alfa per gestire la trasparenza.
    - Infine, l'immagine di sfondo con le sovrapposizioni viene restituita.
    
    In sintesi, questa funzione prende un'immagine e una serie di parametri, esegue varie operazioni di elaborazione delle immagini per rilevare e visualizzare veicoli e linee di corsia, e restituisce l'immagine processata.
    """
    
    global current_offsets, target_offsets, detected_vehicles, destra, sinistra,vehicle_counter
    print("DETECTED VEHICLES",detected_vehicles)
    label_to_image = {
        'car_back': car_back_img,
        'car_front': car_front_img,
        'stop sign': stop_img,
        'motorcycle_back': moto_back,
        'truck_back': truck_back_img
    }

    if confidence <= 0.3 or labels not in label_to_image:
        return

    png_image = label_to_image[labels]
    png_height, png_width, _ = image.shape
    x1, y1, x2, y2 = map(int, coordinates)
    x = max(0, x1)
    y = max(0, y1)
    # distance = min(x2 - x1, 500)
    distance= distance_m
    print("DISTANCEEEEE",distance/250,distance)
    x_offset = (1-(distance / 250)) * png_height
    y = int(x_offset)
    

    #detected_vehicles.sort(key=lambda v: v['distance'])
        
    y -= 150 #sposta in alto le auto
    #x += 20 #sposta a sinistra le auto
    
    
    # if distance < 30 and labels == 'car_back':
    #     x += int(200 * window_scale_factor)
        
    
    if distance <= 50 and labels == 'car_front':
        resized_png = car_front_imgS
        #y += 30
        # if destra:
        #     resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # elif sinistra:
        #     resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), -3, 1.0), (resized_png.shape[1], resized_png.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    elif (distance > 50 and distance <= 150) and labels == 'car_front':
        resized_png = car_front_imgM
        #y += 30
        # if destra:
        #     resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # elif sinistra:
        #     resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), -3, 1.0), (resized_png.shape[1], resized_png.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    elif distance > 150 and labels == 'car_front':
        resized_png = car_front_img
        #y += 30
        
    elif distance <= 50 and labels == 'car_back':
        x -= 30
        
        resized_png = car_back_imgS
        if destra:
            resized_png = cv2.flip(resized_png, 1)
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 5, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 2, 1.0), (resized_png.shape[1], resized_png.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
             resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    elif (distance > 50 and distance <=150)  and labels == 'car_back':
        x += 30
        resized_png = car_back_imgM
        if destra:
            resized_png = cv2.flip(resized_png, 1)
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 5, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 2, 1.0), (resized_png.shape[1], resized_png.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
             resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    elif distance > 150 and labels == 'car_back':
        # y += 30
        # x -= 30
        resized_png = car_back_img
        if destra:
            resized_png = cv2.flip(resized_png, 1)
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 5, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 2, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
             resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    elif distance < 30 and labels == 'motorcycle_back':
        resized_png = moto_back
        y += 30
        # Aggiungi la rotazione per moto_back
        if destra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), -3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
    elif (distance >=30 and distance <=50) and labels == 'motorcycle_back':
        resized_png = moto_back_imgM
        #y += 30
        # Aggiungi la rotazione per moto_back
        if destra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), -3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    elif distance > 50 and labels == 'motorcycle_back':
        resized_png = moto_backS
        # Aggiungi la rotazione per moto_backS
        if destra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), -3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))    
    
    elif distance <= 50 and labels == 'truck_back':
        resized_png = truck_back_imgS
        if destra:
            resized_png = cv2.flip(resized_png, 1)
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 5, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 2, 1.0), (resized_png.shape[1], resized_png.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
             resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    elif (distance > 50 and distance <=150)  and labels == 'truck_back':
        resized_png = truck_back_imgM
        if destra:
            resized_png = cv2.flip(resized_png, 1)
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 5, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 2, 1.0), (resized_png.shape[1], resized_png.shape[0]),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
             resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    elif distance > 150 and labels == 'truck_back':
        resized_png = truck_back_img
        if destra:
            resized_png = cv2.flip(resized_png, 1)
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 5, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif sinistra:
            resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D(((resized_png.shape[1] // 2), (resized_png.shape[0] // 2)), 2, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        else:
             resized_png = cv2.warpAffine(resized_png, cv2.getRotationMatrix2D((resized_png.shape[1] // 2, resized_png.shape[0] // 2), 3, 1.0), (resized_png.shape[1], resized_png.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    else:
        #print("SONO QUI")
        exit(0)
        resized_png = cv2.resize(png_image, (int(100 * window_scale_factor), int(100 * window_scale_factor)), interpolation=cv2.INTER_AREA)
        
    
    # overlapped = False
    # for vehicle_id, (vx1, vy1, vx2, vy2) in detected_vehicles.items():
    #     if (x2 > vx1 and vx2 > x1 and y2 > vy1 and vy2 > y1):
    #         # Sovrapposizione rilevata, non sovrapporre
    #         overlapped = True
    #         print("Sovrapposizione rilevata")
    #         break

    overlay_height, overlay_width, _ = resized_png.shape
    if y + overlay_height > image.shape[0]:
        overlay_height = image.shape[0] - y
    if x + overlay_width > image.shape[1]:
        overlay_width = image.shape[1] - x

    overlay_height = int(overlay_height)
    overlay_width = int(overlay_width)
    overlay = resized_png[:overlay_height, :overlay_width, :3]
    alpha_mask = resized_png[:overlay_height, :overlay_width, 3] / 255.0
    
    overlay_width = max(1, overlay_width)
    overlay_height = max(1, overlay_height)

    # Assicurati che l'area di sovrapposizione non ecceda i limiti dell'immagine di sfondo
    if x < 0: x = 0
    if y < 0: y = 0
    if x + overlay_width > image.shape[1]: overlay_width = image.shape[1] - x
    if y + overlay_height > image.shape[0]: overlay_height = image.shape[0] - y
    
    
    # # Controllo sovrapposizione
    # new_vehicle_position = (x, y, x + overlay_width, y + overlay_height)
    # for vehicle_id, pos in detected_vehicles.items():
    #     if (new_vehicle_position[0] < pos[2] and new_vehicle_position[2] > pos[0] and
    #         new_vehicle_position[1] < pos[3] and new_vehicle_position[3] > pos[1]):
    #         while (new_vehicle_position[0] < pos[2] and new_vehicle_position[2] > pos[0] and
    #                new_vehicle_position[1] < pos[3] and new_vehicle_position[3] > pos[1]):
    #             x += 10  # Spostamento di esempio, da adattare alla tua logica
    #             new_vehicle_position = (x, y, x + overlay_width, y + overlay_height)

    # vehicle_counter += 1
    # detected_vehicles[vehicle_counter] = new_vehicle_position


    if overlay_height > 0 and overlay_width > 0:
        for c in range(3):
            image[y:y + overlay_height, x:x + overlay_width, c] = (
                alpha_mask * overlay[:, :, c] + (1 - alpha_mask) * image[y:y + overlay_height, x:x + overlay_width, c]
            ).astype(np.uint8)
            
        


def overlay_fixed_car_image(image, car_fix, car_fix2, car_fix_curve_left,car_fix_curve_right, window_scale_factor, parameters,car_fix_move,car_fix2_move,car_fix_curve_left_move,car_fix_curve_right_move):
    """
    Funzione responsabile per sovrapporre un'immagine fissa di una macchina su un'altra immagine. Prende in input diversi parametri, tra cui l'immagine di base, diverse immagini di correzione per diverse situazioni stradali, un fattore di scala per la finestra, e altri parametri.

    La funzione inizia controllando il valore del parametro "road_info", che sembra rappresentare le informazioni sulla strada. Se il valore è "Straight" (dritto), viene impostato un flag per indicare che la macchina sta andando dritta e viene selezionata l'immagine di correzione corrispondente. Se invece il valore è "curving to Left" (curva a sinistra) o "curving to Right" (curva a destra), vengono impostati i flag corrispondenti e viene selezionata l'immagine di correzione appropriata.

    Se il valore di "road_info" non corrisponde a nessuna delle situazioni precedenti, viene selezionata un'immagine di correzione di default.

    Successivamente, l'immagine di correzione viene ridimensionata in base al fattore di scala della finestra. Viene quindi calcolata la posizione in cui sovrapporre l'immagine sulla base dell'immagine di input. Se l'immagine di correzione supera i limiti dell'immagine di input, viene ridimensionata di conseguenza.

    Viene quindi creata un'immagine sovrapposta utilizzando una maschera alfa per gestire la trasparenza dell'immagine di correzione. L'immagine sovrapposta viene quindi combinata con l'immagine di input utilizzando una combinazione lineare dei canali di colore.

    Infine, la funzione termina senza restituire alcun valore.

    È importante notare che alcune parti del codice sono state commentate, quindi potrebbero essere state temporaneamente disabilitate o in fase di sviluppo.
    
    """
    
    global destra, sinistra ,use_car_fix
    #print("Sono QUIIII")
    fixed_image=car_fix2
    if parameters["road_info"] == "Straight":
        destra = False
        sinistra = False
        #print("Straight")
        if use_car_fix:
            fixed_image = car_fix
        else:
            fixed_image = car_fix_move
    
    # Alterna il valore della variabile per il prossimo utilizzo
        use_car_fix = not use_car_fix
    elif parameters["road_info"] == "curving to Left":
        destra = False
        sinistra = True
        #print("curving to Left")
        if use_car_fix:
            fixed_image = car_fix_curve_left
        else:
            fixed_image = car_fix_curve_left_move
            
        use_car_fix = not use_car_fix
    elif parameters["road_info"] == "curving to Right":
        #print("curving to Right")
        sinistra = False
        destra = True
        if use_car_fix:
            fixed_image = car_fix_curve_right
        else:
           fixed_image = car_fix_curve_right_move
           
        use_car_fix = not use_car_fix
        
    else:
        if use_car_fix:
            fixed_image = car_fix2
        else:
           fixed_image = car_fix2_move
        
        use_car_fix = not use_car_fix
        

    #fixed_image = cv2.resize(fixed_image, (int(350 * window_scale_factor), int(250 * window_scale_factor)))
    height, width, _ = image.shape
    fixed_height, fixed_width, _ = fixed_image.shape
    x = int((width - fixed_width) // 2)
    y = int(height - fixed_height - 10) 

    if y + fixed_height > height:
        fixed_height = height - y
    if x + fixed_width > width:
        fixed_width = width - x

    overlay = fixed_image[:fixed_height, :fixed_width, :3]
    alpha_mask = fixed_image[:fixed_height, :fixed_width, 3] / 255.0

    for c in range(3):
        image[y:y + fixed_height, x:x + fixed_width, c] = (
            alpha_mask * overlay[:, :, c] + (1 - alpha_mask) * image[y:y + fixed_height, x:x + fixed_width, c]
        ).astype(np.uint8)
    
    # if lane:
    #     add_height, add_width, _ = fixed_image.shape
    #     add_x = int((width - add_width) // 2)
    #     add_y = int((height - add_height) // 2)
        
    #     if add_y + add_height > height:
    #         add_height = height - add_y +10
    #     if add_x + add_width > width:
    #         add_width = width - add_x +20
        
    #     add_overlay = fixed_image[:add_height, :add_width, :3]
    #     add_alpha_mask = fixed_image[:add_height, :add_width, 3] / 255.0
        
    #     for c in range(3):
    #         image[add_y:add_y + add_height, add_x:add_x + add_width, c] = (
    #             add_alpha_mask * add_overlay[:, :, c] + 
    #             (1 - add_alpha_mask) * image[add_y:add_y + add_height, add_x:add_x + add_width, c]
    #         ).astype(np.uint8)
    
    
    
#def process_frame(frame, yolo_model, window_scale_factor, car_fix, car_back_img, car_front_img, stop_img):
def process_frame(params):
    global detected_front_vehicles
    global vehicle_counter
    global detected_vehicles, destra
    
    frame = params.frame_resized
    yolo_model = params.yolo_model
    window_scale_factor = params.window_scale_factor
    car_fix = params.car_fix
    car_fix2 = params.car_fix2
    car_back_img = params.car_back_img
    car_back_imgS = params.car_back_imgS
    car_front_imgS = params.car_front_imgS
    car_front_img = params.car_front_img
    stop_img = params.stop_img
    mtx = params.mtx
    dist = params.dist
    th_sobelx = params.th_sobelx
    th_sobely = params.th_sobely
    th_mag = params.th_mag
    th_dir = params.th_dir
    th_h = params.th_h
    th_l = params.th_l
    th_s = params.th_s
    left_line = params.left_line
    right_line = params.right_line
    focal_length_px = params.focal_length_px
    vehicle_height_m = params.vehicle_height_m
    moto_back_img = params.moto_back_img
    moto_back_imgS = params.moto_back_imgS
    car_fix_curve_left = params.car_fix_curve_left
    car_fix_curve_right = params.car_fix_curve_right
    car_fix_move = params.car_fix_move
    car_back_imgM = params.car_back_imgM
    car_front_imgM = params.car_front_imgM
    moto_back_imgM = params.moto_back_imgM
    car_fix2_move = params.car_fix2_move
    car_fix_curve_left_move = params.car_fix_curve_left_move
    car_fix_curve_right_move = params.car_fix_curve_right_move
    truck_back_img = params.truck_back_img
    truck_back_imgS = params.truck_back_imgS
    truck_back_imgM = params.truck_back_imgM

    
    detected_vehicles.clear()
    vehicle_counter = 0
    


    gray_background = np.ones_like(frame) * 100 #bianco scuro
    #gray_background = np.ones_like(frame) * 50  # Grigio scuro
    
    img, lane,parameters = pipeline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mtx, dist, th_sobelx, th_sobely, th_mag, th_dir, th_h, th_l, th_s, left_line, right_line)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('./output_images/pp1.png', img)
    output = detect_objects(img, yolo_model)

    
    for j in range(len(output)):
        labels = yolo_model.names[int(output[j, 5])]
        coordinates = output[j, :4].tolist()
        confidence = np.round(output[j, 4].item(), 2)

        xmin, ymin, xmax, ymax = map(int, coordinates)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, f'{labels} {confidence}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        # Calcola l'altezza del bounding box in pixel
        bbox_height_px = ymax - ymin

        # Stima la distanza utilizzando la formula di similitudine dei triangoli
        distance_m = (vehicle_height_m * focal_length_px) / bbox_height_px
        distance = min(xmax - xmin, 300)
        distance_text = f'{distance_m:.2f}m'
        #distance_text = f'{distance:.2f}m'
        cv2.putText(img, distance_text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        #detected_vehicles.append({'label': labels, 'distance': distance_m, 'coordinates': coordinates, 'confidence': confidence})
        vehicle_counter = len(detected_vehicles) + 1  # Questo è un esempio semplice usando un ID incrementale

        # Aggiungi il nuovo veicolo al dizionario
        detected_vehicles[vehicle_counter] = (xmin, ymin, xmax, ymax)
        overlay_png(gray_background, coordinates, labels, window_scale_factor, car_back_img, car_back_imgS, car_front_imgS, car_front_img, stop_img, confidence, moto_back_img, moto_back_imgS,distance_m,car_back_imgM,car_front_imgM,moto_back_imgM,truck_back_img,truck_back_imgS,truck_back_imgM)

        
        
        # if labels == 'car_front' and confidence > 0.5:
        #     new_detected_front_vehicles[j] = (xmin, ymin, xmax, ymax)

        # print(f'Object {j + 1} is: {labels}')
        # print(f'Coordinates are: {coordinates}')
        # print(f'Confidence is: {confidence}')
        # print('-------')
        
    # Assumendo che 'frame' sia già definito
    #cv2.imwrite('./output_images/pp1.png', frame)
    
    
    # Controllo del valore di "curvature"
        
    
    overlay_fixed_car_image(gray_background, car_fix, car_fix2, car_fix_curve_left,car_fix_curve_right, window_scale_factor, parameters,car_fix_move,car_fix2_move,car_fix_curve_left_move,car_fix_curve_right_move)

    rows, cols = img.shape[:2]
    #print("Row2, cols2", rows, cols)
    
    return gray_background, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def undistort(img, mtx, dist):
    """ 
    #--------------------
    # undistort image 
    #
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

def pipeline(frame, mtx, dist, th_sobelx, th_sobely, th_mag, th_dir, th_h, th_l, th_s, left_line, right_line):
    identificate_lane_lines = True
    # Correcting for Distortion
    undist_img = undistort(frame, mtx, dist)

    # resize video
    #undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undist_img.shape[:2]
    #cv2.imwrite('./output_images/pp1.png', undist_img)

    combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)

    combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)
    #cv2.imwrite('./output_images/pp2.png', combined_hls)

    combined_result = combine_grad_hls(combined_gradient, combined_hls)
    
    #base
    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 60, 1], [c_cols / 2 +50, 1] #5 abbasa il quadrato se aumetato e lo laza  se diminuito 120 punto in alto a sinsitra
    s_LBot2, s_RBot2 = [110-20, c_rows], [c_cols - 110+70, c_rows]
    
    #autostrada
    # c_rows, c_cols = combined_result.shape[:2]
    # s_LTop2, s_RTop2 = [c_cols / 2 - 10, 10], [c_cols / 2 +50, 10] #5 abbasa il quadrato se aumetato e lo laza  se diminuito 120 punto in alto a sinsitra
    # s_LBot2, s_RBot2 = [110-40, c_rows], [c_cols - 110+50, c_rows]
    
    
    # tornanti
    # c_rows, c_cols = combined_result.shape[:2]
    # s_LTop2, s_RTop2 = [c_cols / 2 - 200, 50], [c_cols / 2 +120, 50] #5 abbasa il quadrato se aumetato e lo laza  se diminuito 120 punto in alto a sinsitra
    # s_LBot2, s_RBot2 = [110-70, c_rows], [c_cols - 110+40, c_rows]

    punti = np.array([[s_LBot2, s_LTop2, s_RTop2, s_RBot2]], dtype=np.int32)
    combined_result2=combined_result
    # Disegna il rettangolo sull'immagine
    cv2.polylines(combined_result2, [punti], isClosed=True, color=(255, 0, 0), thickness=2)
    #cv2.imwrite('./output_images/07_warped_img2.png', combined_result2)
        
    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 300), (550, 300), (550, 720)])
    
    #print("Image shape con tre canali 24", combined_result.shape)
    combined_result = combine_grad_hls(combined_gradient, combined_hls)
    
    warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))
    warp_img2 = warp_img.copy()
    punti2 = np.array([[dst[0], dst[1], dst[2], dst[3]]], dtype=np.int32)
    cv2.polylines(warp_img2, [punti2], isClosed=True, color=(255, 0, 0), thickness=2)
    # cv2.imwrite('./output_images/07_warped_img5.png', warp_img2)
    # cv2.imwrite('./output_images/pp3.png', warp_img)
    cv2.polylines(warp_img, [punti], isClosed=True, color=(255, 0, 0), thickness=2)
    # cv2.imwrite('./output_images/07_warped_img3.png', warp_img)
    
    #print("Image shape con tre canali 24", warp_img.shape)
    mask = np.zeros_like(warp_img)

    # Disegna il rettangolo bianco (o altro colore) esterno
    cv2.fillPoly(mask, [punti2], color=(255, 255, 255))

    # Applica la maschera a warp_img per mantenere solo la regione interna
    masked_img = cv2.bitwise_and(warp_img, mask)
    searching_img = get_lane_lines_img(masked_img, left_line, right_line)
    if left_line.detected and right_line.detected:
        identificate_lane_lines
    else:
        identificate_lane_lines = False
    #cv2.imwrite('./output_images/pp4.png', searching_img)
    

    w_comb_result, w_color_result = illustrate_driving_lane(searching_img, left_line, right_line)
    #cv2.imwrite('./output_images/pp5.png', w_color_result)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    lane_color = np.zeros_like(undist_img)
    lane_color[220:rows - 12, 0:cols] = color_result

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, lane_color, 0.5, 0)

    info_panel, birdeye_view_panel = np.zeros_like(result),  np.zeros_like(result)
    info_panel[5:110, 5:325] = (255, 255, 255)
    birdeye_view_panel[5:110, cols-111:cols-6] = (255, 255, 255)

    info_panel = cv2.addWeighted(result, 1, info_panel, 0.2, 0)
    birdeye_view_panel = cv2.addWeighted(info_panel, 1, birdeye_view_panel, 0.2, 0)
    road_map = illustrate_driving_lane_with_topdownview(w_color_result, left_line, right_line)
    birdeye_view_panel[10:105, cols-106:cols-11] = road_map
    birdeye_view_panel, parameters = illustrate_info_panel(birdeye_view_panel, left_line, right_line)

    return birdeye_view_panel,identificate_lane_lines,parameters

def add_performance_measure(frame_number, processing_time):
    global performance_data
    new_row = pd.DataFrame({'Frame': [frame_number], 'Processing Time': [processing_time]})
    performance_data = pd.concat([performance_data, new_row], ignore_index=True)


def main():
    yolo_repo_path = 'yolo/yolov5-master'
    model_path = 'yolo/yolov5_vehicle_oriented.pt'

    yolo_model = load_yolo_model(yolo_repo_path, model_path)

    video_path = 'test_video/project_video.mp4'
    #video_path = 'C:/Users/luigi/OneDrive/Documenti/Lane_detect/advanced-lane-detection-for-self-driving-cars-master/harder_challenge_video.mp4'
    #video_path = 'test_video/Strada.mp4'
    #video_path = 'C:/Users/luigi/OneDrive/Documenti/Lane_detect/advanced-lane-detection-for-self-driving-cars-master/challenge_video.mp4'
    #video_path = 'videoStrada2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        #print("Errore nell'apertura del video.")
        return
    
    #video_path=None

    # Definizione delle dimensioni ridotte delle finestre di visualizzazione
    window_scale_factor = 1/2  # Riduzione del 50%

    # Creiamo un ThreadPoolExecutor con un numero arbitrario di thread
    executor = ThreadPoolExecutor(max_workers=4)
    
    
    car_back_img = cv2.imread('img_Project/car_back2.png', cv2.IMREAD_UNCHANGED)
    car_front_img = cv2.imread('img_Project/car_front2.png', cv2.IMREAD_UNCHANGED)
    stop_img = cv2.imread('img_Project/stop.png', cv2.IMREAD_UNCHANGED)
    moto_back = cv2.imread('img_Project/moto_back.png', cv2.IMREAD_UNCHANGED)
    truck_back = cv2.imread('img_Project/truck_back.png', cv2.IMREAD_UNCHANGED)
    
    car_back_img_original = car_back_img.copy()
    car_front_img_original = car_front_img.copy()
    truck_back_img_original = truck_back.copy()

    # Ridimensiona direttamente dall'immagine originale
    car_back_img = cv2.resize(car_back_img_original, (int(50 * window_scale_factor), int(50 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    car_back_imgM = cv2.resize(car_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    car_back_imgS = cv2.resize(car_back_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)), interpolation=cv2.INTER_AREA)

    car_front_img = cv2.resize(car_front_img_original, (int(50 * window_scale_factor), int(50 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    car_front_imgM = cv2.resize(car_front_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    car_front_imgS = cv2.resize(car_front_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    moto_back_img = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    moto_back_imgM = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    moto_back_imgS = cv2.resize(moto_back, (int(300 * window_scale_factor), int(300 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    truck_back_img = cv2.resize(truck_back_img_original, (int(50 * window_scale_factor), int(50 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    truck_back_imgM = cv2.resize(truck_back_img_original, (int(130 * window_scale_factor), int(130 * window_scale_factor)), interpolation=cv2.INTER_AREA)
    truck_back_imgS = cv2.resize(truck_back_img_original, (int(170 * window_scale_factor), int(170 * window_scale_factor)), interpolation=cv2.INTER_AREA)

   
    car_fix = cv2.imread('img_Project/carline.png', cv2.IMREAD_UNCHANGED)
    car_fix = cv2.resize(car_fix, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)

    car_fix_move = cv2.imread('img_Project/carline2.png', cv2.IMREAD_UNCHANGED)
    car_fix_move = cv2.resize(car_fix_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    car_fix2 = cv2.imread('img_Project/no_carline.png', cv2.IMREAD_UNCHANGED)
    car_fix2 = cv2.resize(car_fix2, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    car_fix2_move = cv2.imread('img_Project/no_carline2.png', cv2.IMREAD_UNCHANGED)
    car_fix2_move = cv2.resize(car_fix2_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    car_fix_curve_left = cv2.imread('img_Project/carline_left.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_left = cv2.resize(car_fix_curve_left, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)

    car_fix_curve_left_move = cv2.imread('img_Project/carline_left2.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_left_move = cv2.resize(car_fix_curve_left_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    car_fix_curve_right = cv2.imread('img_Project/carline_right.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_right = cv2.resize(car_fix_curve_right, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    car_fix_curve_right_move = cv2.imread('img_Project/carline_right2.png', cv2.IMREAD_UNCHANGED)
    car_fix_curve_right_move = cv2.resize(car_fix_curve_right_move, (int(450 * window_scale_factor), int(450 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    
    

    #car_back_img=cv2.resize(car_back_img, (int(100 * window_scale_factor), int(100 * window_scale_factor)),interpolation=cv2.INTER_AREA)
    
    mtx, dist = calib()
    # th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    # th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)
    th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)
    left_line = Line()
    right_line = Line()
    
    # Parametri della telecamera
    focal_length_px = mtx[1, 1]  # Distanza focale in pixel (f_y)
    vehicle_height_m = 1.5  # Altezza media del veicolo in metri
    # Assicurati che l'immagine sia stata caricata correttamente
    
    # Inizializzazione della finestra Tkinter in un thread separato
    tkinter_thread = Thread(target=run_tkinter)
    tkinter_thread.start()
    frame_number = 0
    #tracker = EmissionsTracker()
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    tracker.start()
    while True:
        
        if video_path != None:
            ret, frame = cap.read()

            if not ret:
                break
        else:
            frame = cv2.imread("runs/detect/predict5/t.png")
        window_scale_factor= 1/2
        window_scale_factor= 1/2
        # Ridimensioniamo il frame letto per corrispondere alle dimensioni ridotte
        add_to_frames_to_save(frame)
        frame_resized = cv2.resize(frame, None, fx=window_scale_factor, fy=window_scale_factor,interpolation=cv2.INTER_AREA)
        
        # start_time = time.time()
        
        #add_to_frames_to_save(frame)
        
       

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #   Tune Parameters for different inputs        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # camera matrix & distortion coefficient
        # Assegnamento manuale dei valori di mtx e dist
        # mtx = np.array([
        #     [1.15663033e+03, 0.00000000e+00, 6.69042373e+02],
        #     [0.00000000e+00, 1.15169376e+03, 3.88133397e+02],
        #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        # ])

        # dist = np.array([-0.2315702, -0.12000305, -0.00118318, 0.00023296, 0.15639731])
        
        params = ProcessingParams(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), yolo_model, window_scale_factor, car_fix, car_fix2, car_back_img,
                                  car_back_imgS, car_front_imgS, car_front_img, stop_img, mtx, dist, th_sobelx, th_sobely, th_mag, th_dir, th_h,
                                  th_l, th_s, left_line, right_line, focal_length_px, vehicle_height_m, moto_back_img,
                                  moto_back_imgS, car_fix_curve_left, car_fix_curve_right, car_fix_move, car_back_imgM, car_front_imgM, moto_back_imgM,
                                  car_fix2_move, car_fix_curve_left_move, car_fix_curve_right_move,
                                  truck_back_img, truck_back_imgM, truck_back_imgS)
        
        future = executor.submit(process_frame, params)
        
        # Eseguiamo il processamento di ogni frame utilizzando executor.submit
        # future = executor.submit(process_frame, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB), yolo_model, window_scale_factor, car_fix,car_fix2, car_back_img,
        # car_back_imgS,car_front_imgS, car_front_img, stop_img, mtx, dist, th_sobelx, th_sobely, th_mag, th_dir, th_h,
        # th_l, th_s, left_line, right_line,focal_length_px, vehicle_height_m,moto_back_img,
        # moto_back_imgS,car_fix_curve_left,car_fix_curve_right,car_fix_move,car_back_imgM,car_front_imgM,moto_back_imgM,
        # car_fix2_move,car_fix_curve_left_move,car_fix_curve_right_move,
        # truck_back_img, truck_back_imgM, truck_back_imgS)
        #future = executor.submit(process_frame, frame_resized, yolo_model, window_scale_factor, car_fix, car_back_img, car_front_img, stop_img)
        # Recuperiamo il risultato del processamento del frame
        gray_background, img = future.result()

        # Creiamo l'immagine concatenata
        concatenated_img = np.concatenate((gray_background, img), axis=1)
        
        frames_to_save.append(concatenated_img.copy())
        
        

        cv2.namedWindow('Object Detection Overlay', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection Overlay', int(2 * img.shape[1]), int(img.shape[0]))

        cv2.imshow('Object Detection Overlay', concatenated_img)
        
        # end_time = time.time()
        # processing_time = end_time - start_time
        
        # # Aggiungi la misura prestazionale
        # add_performance_measure(frame_number, processing_time)
        
        # frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    #performance_data.to_csv('performance_data_all.csv', index=False)
    #print(performance_data)
    cap.release()
    cv2.destroyAllWindows()
    
    # Ferma il tracker di CodeCarbon
    tracker.stop()
    

def add_to_frames_to_save(frame):
    global frames_to_save
    max_frames = 50
    
    # Calcola quanti frame devono essere eliminati
    excess_frames = len(frames_to_save) - max_frames
    if excess_frames > 0:
        # Elimina i frame in eccesso
        del frames_to_save[:excess_frames]

def save_text(text_entry_widget):
    global frames_to_save
    text_to_save = text_entry_widget.get("1.0", "end-1c")  # Ottieni il testo inserito dall'utente
    if text_to_save.strip():  # Verifica se il testo non è vuoto
        # Salva il testo in un file di testo
        with open("testo_salvato.txt", "w") as file:
            file.write(text_to_save)

        # Salva gli ultimi 50 frame concatenati come immagini PNG
        frames_folder = "frame_video"
        os.makedirs(frames_folder, exist_ok=True)
        for idx, frame in enumerate(frames_to_save):
            filename = os.path.join(frames_folder, f"frame_{idx}.png")
            cv2.imwrite(filename, frame)

        messagebox.showinfo("Salvataggio completato", "Il testo e i frame sono stati salvati correttamente!")
    else:
        messagebox.showwarning("Nessun testo", "Inserisci del testo prima di salvare.")

def run_tkinter():
    global text_save_window
    text_save_window = tk.Tk()
    text_save_window.title("Salvataggio di testo e immagini")

    # Frame per il testo
    text_frame = tk.Frame(text_save_window, padx=20, pady=20)
    text_frame.pack()

    text_label = tk.Label(text_frame, text="Inserisci il testo da salvare:")
    text_label.pack()

    text_entry = tk.Text(text_frame, height=10, width=50)
    text_entry.pack(pady=10)

    # Bottone Salva
    save_button = tk.Button(text_frame, text="Salva", command=lambda: save_text(text_entry))
    save_button.pack()

    text_save_window.mainloop()


if __name__ == "__main__":
    main()
