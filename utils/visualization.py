import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


COLORMAP = {
        'Background': (125, 125, 125),          # white
        'background': (125, 125, 125),          # white
        'unanotated': (125, 125, 125),          # white
        'Unanotated': (125, 125, 125),          # white
        'Lymph node': (229, 100, 84),           # orange
        'lymph_node': (229, 100, 84),           # orange
        'Tumor deposits': (212, 185, 60),       # dark yellow
        'tumor_deposit_or_primary_tumor': (212, 185, 60),       # dark yellow
        'Primary tumor': (54, 90, 113),         # blue
        'Primary tissue': (0, 124, 169),        # cyan
        'healthy_primary': (0, 124, 169),        # cyan
        'Ink': (11, 72, 205),                   # dark blue
        'Vessels': (106, 29, 125),              # purple
        'Metastasis': (117, 173, 81),           # light green
        'Necrosis': (50, 50, 50),               # grey
        'Connective tissue': (250, 71, 102),    # red
        'Folds': (73, 103, 40),                 # dark green
        'Fat tissue': (255, 255, 153),          # light yellow
        'Mucin': (220, 220, 220),               # light grey
        'mucin': (220, 220, 220),               # light grey
        'Slide edge': (48, 213, 200),           # turquoise
        'Training region': (0, 0, 0),           # black

        'other': (125, 125, 125),               # white
        'tumor':(117, 173, 81),                 # light green
        'Tumor':(117, 173, 81),                 # light green
        'stroma':(250, 71, 102),                # red
        'necrosis_or_debris':(50, 50, 50),      # grey
        'fat': (255, 255, 153),                 # light yellow
        'mucoid_material': (220, 220, 220),     # light grey
        'blood': (106, 29, 125),                # purple
        'lymphocytic_infiltrate': (0, 0, 128),  # navy blue 
        'muscle': (128, 0, 0),                   # maroon
        'nerve': (128, 128, 0),                   # olive
        'normal_mucosa': (0, 124, 169),        # cyan
        'vessel': (106, 29, 125),              # purple

        "Fat": (255, 255, 153),                 # light yellow,
        "Normal Mucosa": (0, 124, 169),         # cyan
        "Lymphoid tissue": (0, 0, 128),         # maroon
        "Stroma": (255, 182, 193),              # pink
        "Mucous": (220, 220, 220),              # light grey
        "Necrosis/debris": (50, 50, 50),        # grey
        "Muscle": (128, 0, 0),                  # maroon
        "Muscle/vessel": (128, 0, 0),           # maroon
        "muscle/vessel": (128, 0, 0),           # maroon
        "Nerve": (128, 128, 0),                 # olive
        "Blood": (250, 71, 102)                 # red
    }


def save_overlay(out_path:str, wsi_path:str, mask:np.ndarray, level:int, level_downsampling:int, read_origin:tuple, label2id:dict,downsizing_factor:int=1, colormap:dict=COLORMAP, ):
    # get level from mpp
    slide = openslide.open_slide(wsi_path)
    # get dimensions of H&E
    w, h = slide.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH], slide.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT]
    newDim = (int(int(w)/level_downsampling),int(int(h)/level_downsampling))

    slide = np.array(slide.read_region(read_origin, level, newDim))  
    slide[slide[:,:,3] == 0] = 255
    slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)
    if downsizing_factor > 1:
        slide = cv2.resize(slide, (slide.shape[1]//downsizing_factor, slide.shape[0]//downsizing_factor), interpolation=cv2.INTER_NEAREST)
    # if different shape, resize mask
    if mask.shape != slide.shape[:2]:
        mask = cv2.resize(mask, (slide.shape[1], slide.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((5,5), np.uint8)  # Define structuring element
    # Closing (fills small holes and smooths edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # create contours (with holes)
    # for label, id in label2id.items():
    #     if id not in mask:
    #         continue
    #     aux_mask = (mask == id).astype(np.uint8)
    #     # erode before finding contours to avoid overlapping contours
    #     aux_mask = cv2.erode(aux_mask, kernel, iterations=2)
    #     contours, __ = cv2.findContours(aux_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #     # draw contours
    #     for contour in contours:
    #         cv2.drawContours(slide, contour, -1, colormap[label], thickness=4)
    
    # slide = cv2.cvtColor(slide, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(out_path, slide)

    # now create png for the mask
    id2label = {v: k for k, v in label2id.items()}
    out_path = out_path.replace('.png', '_mask.png')
    png = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    present = []
    for i in id2label.keys():
        if i in mask:
            png[mask == i] = colormap[id2label[i]]
            present.append(i)
    # create figure
    patches = [mpatches.Patch(color=np.array(colormap[id2label[i]])/255, label=id2label[i]) for i in id2label.keys() if i in present]
    plt.imshow(png)
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0) 
    plt.close()   