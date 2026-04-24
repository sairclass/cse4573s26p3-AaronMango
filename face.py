'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    imgNUMPY = img.permute(1, 2, 0).contiguous().cpu().numpy().copy()
    '''print("----------------------------")
    print(type(imgNUMPY))
    print(imgNUMPY.shape)
    print(imgNUMPY.dtype)
    print(type(imgNUMPY[0,0,0]))
    print(imgNUMPY.flags)
    print(imgNUMPY.min(), imgNUMPY.max())
    print("----------------------------")'''
    faceLocations = face_recognition.face_locations(imgNUMPY)

    retFaces = []
    for face in faceLocations:
        x = float(face[3])
        y = float(face[0])
        w = float(face[1] - face[3])
        h = float(face[2] - face[0])
        newFace = [x,y,w,h]
        retFaces.append(newFace)
    return retFaces



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    names = []
    encodings = []
    for img in imgs:
        imgNUMPY = imgs[img].permute(1, 2, 0).contiguous().cpu().numpy().copy()
        faceLocation = face_recognition.face_locations(imgNUMPY)
        encoding = face_recognition.face_encodings(imgNUMPY,faceLocation)[0]
        encoding = torch.tensor(encoding,dtype=torch.float32)

        encoding = encoding / torch.norm(encoding) #normalizing for better results hopefully (IT worked lets fricken go)

        names.append(img)
        encodings.append(encoding)
    

    randomIndices = torch.randperm(len(encodings))[:K]
    stackedEncodings = torch.stack(encodings)
    centroids = stackedEncodings[randomIndices]

    maxIterations = 5000
    for _ in range(maxIterations):
        groupings = []
        for _ in range(K):
            groupings.append([])

        for encoding in encodings:
            minDist = (-1,float('inf'))
            for i, centroid in enumerate(centroids):
                dist = torch.norm(encoding - centroid)
                if dist < minDist[1]:
                    minDist = (i,dist)
            groupings[minDist[0]].append(encoding)

        newCentroids = []
        movementOfCentroids = 0
        for i, grouping in enumerate(groupings):
            if len(grouping) != 0:
                newCentroids.append(torch.mean(torch.stack(grouping), dim=0))
            else:
                newCentroids.append(centroids[i])
            movementOfCentroids += torch.norm(centroids[i] - newCentroids[i])

        newCentroids = torch.stack(newCentroids)
        threshold = 1e-4
        if movementOfCentroids < threshold:
            centroids = newCentroids
            break
        else:
            centroids = newCentroids

    clusters = []
    for i in range(K):
        clusters.append([])

    for i, encoding in enumerate(encodings):
        minDist = (-1,float('inf'))
        for j, centroid in enumerate(centroids):
            dist = torch.norm(encoding - centroid)
            if dist < minDist[1]:
                minDist = (j,dist)
        clusters[minDist[0]].append(names[i])

    return clusters

'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
