# some functions for creation and rendering of 3d objects were borrowed/inspired from 
# Juan Gallostra Acín's repository - https://github.com/juangallostra/augmented-reality 

'''
Copyright (c) 2018 Juan Gallostra Acín

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER I
N AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

#contains code to the .obj file and augment the object

import cv2
import copy
import numpy as np
import time

class three_d_object:
    def __init__(self, filename_obj, filename_texture, color_fixed = False):
        self.texture = cv2.imread(filename_texture)
        self.vertices = []
        self.faces = []
        #each face is a list of [lis_vertices, lis_texcoords, color]
        self.texcoords = []
        self.time = 0

        self.position = np.array([[1,0,0,400], [0,1,0,250], [0,0,1,200], [0,0,0,1]], np.float64)

        for line in open(filename_obj, "r"):
            if line.startswith('#'): 
                #it's a comment, ignore 
                continue

            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                #vertex description (x, y, z)
                v = [float(a) for a in values[1:4] ]
                self.vertices.append(v)

            elif values[0] == 'vt':
                #texture coordinate (u, v)
                self.texcoords.append([float(a) for a in values[1:3] ])

            elif values[0] == 'f':
                #face description 
                face_vertices = []
                face_texcoords = []
                for v in values[1:]:
                    w = v.split('/')
                    face_vertices.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        face_texcoords.append(int(w[1]))
                    else:
                        color_fixed = True
                        face_texcoords.append(0)
                self.faces.append([face_vertices, face_texcoords])


        for f in self.faces:
            if not color_fixed:
                f.append(three_d_object.decide_face_color(f[-1], self.texture, self.texcoords))
            else:
                f.append((255, 0, 0)) #default color

        # cv2.imwrite('texture_marked.png', self.texture)

    def move(self, qtd):
        self.position[0, 3] = self.position[0, 3] + qtd

    def decide_face_color(hex_color, texture, textures):
        #doesnt use proper texture
        #takes the color at the mean of the texture coords

        h, w, _ = texture.shape
        col = np.zeros(3)
        coord = np.zeros(2)
        all_us = []
        all_vs = []

        for i in hex_color:
            t = textures[i - 1]
            coord = np.array([t[0], t[1]])
            u , v = int(w*(t[0]) - 0.0001), int(h*(1-t[1])- 0.0001)
            all_us.append(u)
            all_vs.append(v)

        u = int(sum(all_us)/len(all_us))
        v = int(sum(all_vs)/len(all_vs))

        # all_us.append(all_us[0])
        # all_vs.append(all_vs[0])
        # for i in range(len(all_us) - 1):
        #     texture = cv2.line(texture, (all_us[i], all_vs[i]), (all_us[i + 1], all_vs[i + 1]), (0,0,255), 2)
        #     pass    

        col = np.uint8(texture[v, u])
        col = [int(a) for a in col]
        col = tuple(col)
        return (col)


def removeHomogenous(x):
    y = []
    for i in x:
        i = i / i[-1]
        i = i[0:-1]
        i = [int(x) for x in i]
        y.append(i)

    return np.array(y)

def insertHomogenous(x):
    l, c = x.shape
    c = c+1
    temp = np.ones((l, c))
    temp[:, :3] = x
    return temp


cube = three_d_object('data/cube/cube.obj', 'data/Hulk/hulk.png', color_fixed=True)

bullet1 = None
bullet2 = None
dieTime = 10000

def augment1(img, obj, projection, h, w, shoot, t2, scale = 1):
    vertices = obj.vertices

    a = np.array([[0,0,0,1], [w, 0, 0, 1],  [w,h,0,1],  [0, h, 0, 1]], np.float64 )
    imgpts = np.dot(projection, a.T).T
    imgpts = removeHomogenous(imgpts)

    cv2.fillConvexPoly(img, imgpts, (0,0,0))
    drawBullet1(img, projection, shoot)

    for face in obj.faces:
        face_vertices = face[0]
        
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = scale*points
        points = insertHomogenous(points)
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1], p[3]] for p in points]) #shifted to centre 

        global bullet2
        if bullet2 is not None and t2 is not None:
            opos = np.array([0,0,0,1])
            opos = np.dot(projection, opos.T).T
            bpos = np.array([0,0,0,1])
            bpos = np.dot(t2, np.dot(bullet2.position, bpos.T)).T

            opos = opos/opos[2]
            bpos = bpos/bpos[2]
            d = np.linalg.norm(bpos-opos)
            # print(d)
            
            if d < 150:
                print(f"hit b2 {time.time()}")
                bullet2 = None
            # print(f"bpos: {bpos}")
            # print(f"opos: {opos}")

        dst = np.dot(projection, points.T).T
        
        imgpts = removeHomogenous(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])
        
    return img

def augment2(img, obj, projection, h, w, shoot, t1, scale = 1):
    vertices = obj.vertices

    a = np.array([[0,0,0,1], [w, 0, 0, 1],  [w,h,0,1],  [0, h, 0, 1]], np.float64 )
    imgpts = np.dot(projection, a.T).T
    imgpts = removeHomogenous(imgpts)

    cv2.fillConvexPoly(img, imgpts, (0,0,0))
    drawBullet2(img, projection, shoot)

    for face in obj.faces:
        face_vertices = face[0]
        
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = scale*points
        points = insertHomogenous(points)
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1], p[3]] for p in points]) #shifted to centre 

        global bullet1
        if bullet1 is not None and t1 is not None:
            opos = np.array([0,0,0,1])
            opos = np.dot(projection, opos.T).T
            bpos = np.array([0,0,0,1])
            bpos = np.dot(t1, np.dot(bullet1.position, bpos.T)).T

            opos = opos/opos[2]
            bpos = bpos/bpos[2]
            d = np.linalg.norm(bpos-opos)
            # print(d)
            if d < 150:
                print(f"hit b1 {time.time()}")
                bullet1 = None
            # print(f"bpos: {bpos}")
            # print(f"opos: {opos}")
            

        dst = np.dot(projection, points.T).T
        
        imgpts = removeHomogenous(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])
        
    return img


def drawBullet1(img, projection, shoot):
    global bullet1
    if bullet1 is None and shoot != 0:
        bullet1 = copy.deepcopy(cube)
        bullet1.time = time.time()
    
    if bullet1 is not None:
        passedTime = time.time() - bullet1.time
        bullet1.move(passedTime*500)
        if passedTime > 3:
            bullet1 = None
            return
        
        vertices = bullet1.vertices
        for face in bullet1.faces:
            face_vertices = face[0]
            
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = 20*points
            points = insertHomogenous(points)
            points = np.array([[p[0], p[1], p[2], p[3]] for p in points]) #shifted to centre 

            dst = np.dot(bullet1.position, points.T).T
            dst = np.dot(projection, dst.T).T
            
            imgpts = removeHomogenous(dst)
            cv2.fillConvexPoly(img, imgpts, face[-1])


def drawBullet2(img, projection, shoot):
    global bullet2
    if bullet2 is None and shoot != 0:
        bullet2 = copy.deepcopy(cube)
        bullet2.time = time.time()

    if bullet2 is not None:
        passedTime = time.time() - bullet2.time
        bullet2.move(passedTime*2000)
        if passedTime > 3:
            bullet2 = None
            return
        
        vertices = bullet2.vertices
        for face in bullet2.faces:
            face_vertices = face[0]
            
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = 20*points
            points = insertHomogenous(points)
            points = np.array([[p[0], p[1], p[2], p[3]] for p in points]) #shifted to centre 

            dst = np.dot(bullet2.position, points.T).T
            dst = np.dot(projection, dst.T).T
            
            imgpts = removeHomogenous(dst)
            cv2.fillConvexPoly(img, imgpts, face[-1])
