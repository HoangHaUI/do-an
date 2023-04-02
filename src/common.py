import cv2
import numpy as np

def get_coord(img_path :str, txt_path :str):
    # Lay toa do doi tuong 39
    img = cv2.imread(img_path)
    dh, dw, _ = img.shape
    fl = open(txt_path, 'r')
    data = fl.readlines()   
    fl.close()
    list_coord_all = list()

    for dt in data:

        # Split string to float
        _, x, y, w, h, acc = map(float, dt.split(' '))
        if acc < 0.7:
            return

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        b_nap = (int)((b-t)/8) + t
        t_nhan = (int)((b-t)/8) + t
       
        img_nap = img[t:b_nap, l:r]
        img_nhan = img[t_nhan:b, l:r]

        cv2.imwrite('./runs/nap.jpg', img_nap)
        cv2.imwrite('./rums/nhan.jpg', img_nhan)
        list_coord = list()
        list_coord.append([t,b_nap, l,r])
        list_coord.append([t_nhan,b, l,r])
        list_coord_all.append(list_coord)
    return list_coord_all



# Lấy tọa độ của nắp và nhãn trên từng ảnh 
def get_crop_image(classes, coords):
    # coord : xyxy 
    nap_coords = []
    nhan_coords = []
    classes = np.asarray(classes).flatten()
    coords = coords[0]
    coord_bottle = []
    for i in range(len(classes)):
        if classes[i] == 39: # Nhan la bottle 
            coord = coords[i] # Toa do cua bottle
    
            [x1,y1, x2,y2] = coord
            coord_bottle.append([x1,y1, x2,y2])
            y2_nap = y1 + (int)((y2-y1)/7)
            y1_nhan = y2_nap
            nap_coord = [x1,y1,x2,y1_nhan]
            nhan_coord = [x1,y1_nhan, x2,y2]
            nap_coords.append(nap_coord)
            nhan_coords.append(nhan_coord)

    return np.asarray(nap_coords, dtype=int),np.asarray(nhan_coords, dtype=int),np.asarray(coord_bottle, dtype=int)
