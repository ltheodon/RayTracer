import numpy as np
import matplotlib.pyplot as plt
import fxaa






def normalize(x):
    if np.linalg.norm(x) < 1.e-16:
        return x
    x /= np.linalg.norm(x)
    return x

def intersect_plane(l, l0, n, p0):
    denom = np.dot(l, n)
    if np.abs(denom) < 1e-6:
        return np.inf
    if denom < 0:
        return np.inf
    d = np.dot(p0-l0, n) / denom
    return d

def intersect_sphere(O, D, S, R):
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect_square(l, l0, n, p0, length):
    d = intersect_plane(l, l0, n, p0)
    if d == np.inf:
        return d
    M = l0 + d*l
    if any(x > length/2 for x in list(map(abs,p0 - M))):
        return np.inf
    else:
        return d



def intersect_cube(l, l0, obj):
    dFace = np.inf
    N = np.array([0., 0., 0.])
    for face in obj['face']:
        d = intersect_square(l, l0, face['normal'], face['position'], face['length'])
        if d < dFace:
            dFace = d
            M = l0 + dFace*l
            N = get_normal(face, M, l, l0)
    return N, dFace

    

def sphere(position, radius, color, reflection=.0, transparent=.0):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius),
        color=np.array(color), diffuse_c=.75, specular_c=.5, reflection=reflection, transparent=min(0.99,transparent))

def plane(position, normal, color, reflection=.0):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=color, diffuse_c=.75, specular_c=.3, reflection=reflection, transparent=0.)

def square(position, normal, length, color):
    return dict(type='square', position=np.array(position), 
        length=length,
        normal=np.array(normal),
        color=np.array(color), diffuse_c=.75, specular_c=.3, reflection=.0, transparent=0.)


def cube(position, normal_1, normal_2, length, color):
    n3 = np.cross(np.array(normal_1),np.array(normal_2))
    return dict(type='cube', position=np.array(position), 
        length=length,
        normal_1=np.array(normal_1),
        normal_2=np.array(normal_2),
        normal_3=n3,
        face=[square(np.array(position) - length*np.array(normal_1)/2, normal_1, length, color),
            square(np.array(position) + length*np.array(normal_1)/2, [-x for x in normal_1], length, color),
            square(np.array(position) - length*np.array(normal_2)/2, normal_2, length, color),
            square(np.array(position) + length*np.array(normal_2)/2, [-x for x in normal_2], length, color),
            square(np.array(position) - length*np.array(n3)/2, n3, length, color),
            square(np.array(position) + length*np.array(n3)/2, [-x for x in n3], length, color)],
        color=np.array(color), diffuse_c=.75, specular_c=.3, reflection=0., transparent=0.)





def intersect(l, l0, obj):
    if obj['type'] == 'plane':
        return intersect_plane(l, l0, obj['normal'], obj['position'])
    if obj['type'] == 'square':
        return intersect_square(l, l0, obj['normal'], obj['position'], obj['length'])
    if obj['type'] == 'cube':
        N, inter = intersect_cube(l, l0, obj)
        return inter
    elif obj['type'] == 'sphere':
        return intersect_sphere(l0, l, obj['position'], obj['radius'])

def get_normal(obj, M, l = [0.,0.,0.], l0 = [0.,0.,0.]):
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'square':
        N = obj['normal']
    elif obj['type'] == 'cube':
        N, inter = intersect_cube(l, l0, obj)
        return N
    return N

def get_color(obj, M):
    color = obj['color']
    return color


def filterShadow(a,b):
    return a < b

def trace(O, ray, rec=False, nbrec=0):
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(ray, O, obj)
        if t_obj < t:
            t, obj_id = t_obj, i
    #print(t)
    if t == np.inf:
        return
    # En cas d'intersection
    obj = scene[obj_id]
    # Calcul des coordonnées du point d'intersection
    M = O + t*ray
    N = get_normal(obj, M, ray, O)
    color = get_color(obj, M)
    #print(color)
    toLight = normalize(Light - M)
    toOrigin = normalize(O - M)

    shadow_mod = 1
    l = [intersect(toLight, M + N * .0001, obj_shadow)*(1+np.sqrt(obj_shadow['transparent'])) 
            for k, obj_shadow in enumerate(scene) if k != obj_id]
    shadowList = list(map(abs,l))
    LightShadow = np.linalg.norm(Light - M)
    if l and min(shadowList) < LightShadow:
        f = list(filter(lambda x:filterShadow(x,LightShadow),shadowList))
        #InvSquare = [e/(n+1)**2 for (n,e) in enumerate(f)]
        for s in f:
            shadow_mod *= s/LightShadow
        #shadow_mod = sum(InvSquare)/LightShadow/len(f)
        #shadow_mod = min(list(map(abs,l)))/np.linalg.norm(Light - M)
        #print(shadow_mod)
    # Calcul des couleurs
    trans = obj['transparent']*np.sqrt(abs(np.dot(N, ray)))
    col = ambient * color * (1-trans)
    # Lambert (diffuse).
    col += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toLight),0) * color * (1-trans)

    # Blinn-Phong (specular).
    col += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toLight + toOrigin)),0) ** specular_k * color_light

    for i in range(0,1):
        if obj.get('reflection', 0.) > 0. and rec and nbrec < 6:
            #print(nbrec % 2)
            #print(nbrec)
            if nbrec % 2 == 1:
                refRay = normalize(-ray + 2 * np.dot(ray, N) * N)
                refTr = trace(M - 0.001*N,refRay,False,nbrec)
            else:
                refRay = normalize(ray - 2 * np.dot(ray, N) * N)
                refTr = trace(M + 0.001*N,refRay,False,nbrec)
            #refTr = trace(M + 0.001*N,refRay,True,nbrec+1)
            if not refTr:
                break
            objr, Mr, Nr, col_r, tranR = refTr
            col += col_r*obj.get('reflection', 0.)

    if trans > 0.:
        tr = trace(M + 0.001*ray,ray,True,nbrec+1)
        objt, Mt, Nt, col_ray, trans_r = tr
        #col += np.clip(col_ray, 0, 1)*trans
        col += col_ray*trans
    return obj, M, N, (col * np.sqrt(np.sqrt(shadow_mod))), trans





#color_plane = np.array([0.7, 0.01, 0.02])
color_plane = np.array([0.35, 0.25, 0.9])
color_red = np.array([1., 0.01, 0.01])
color_orange = np.array([0.88, 0.43, 0.08])
color_blue = np.array([0., 0.00, 1.])
color_green = np.array([0.1, 0.95, 0.1])
color_yellow = np.array([1, 0.9, 0.1])
color_black = np.array([0., 0., 0.])
color_white = np.array([1., 1., 1.])
color_gray = np.array([0.25, 0.25, 0.25])


Light = np.array([6.5, 5., -0.])
color_light = np.ones(3)

ambient = .35
diffuse_c = 1.
specular_c = 1.
specular_k = 50

scene = []
scene.append(plane([0,0,25], [0,0,1], color_plane, 0.5))
scene.append(plane([0,0,-0], [0,0,-1], color_plane, 0.0))
scene.append(plane([0,12,0], [0,1,0], color_plane, 0.0))
scene.append(plane([0,0,10], [0,-1,0], color_plane, 0.0))
scene.append(plane([8,0,10], [1,0,0], color_plane))
scene.append(plane([0,0,10], [-1,0,0], color_plane))
scene.append(sphere([0.5, 3, 4.], .5, color_red))
scene.append(sphere([1, 6.0, 5.5], 1., color_green, 0.85, 0.99))
scene.append(sphere([1.5, 3.5, 7.0], 1.2, color_yellow, 0.85, 0.5))
scene.append(sphere([2.2, 4.8, 5.0], 0.6, color_blue))
scene.append(sphere([2.5, 8.0, 2.5], 2.5, color_white, 0.25, 1.))
scene.append(sphere([4.5, 2.5, 4.5], .7, color_orange, 0.))
scene.append(sphere([3.5, 5, 11.5], 3.5, color_black, 1., 0.))
scene.append(sphere([1.0, 9, 10.0], 1.0, color_yellow, 0.5, 0.))

#scene.append(square([4,4,5], [0,0,1], 3., color_red))

# Ne fonctionne pas (mauvais calcul des faces)
#scene.append(cube([2,2,3], [1/1.41,1/1.41,0], [-1/1.41,1/1.41,0], 1, color_red))
#scene.append(cube([2,9,6], [1,0,0], [0,1,0], 1, color_red))


Screen = [8,12]
Camera = np.array([Screen[0]/2,Screen[1]/2,-28])

depth_max = 4 # Profondeur max de réflexion

w = 800
h = w*3//4

img = np.zeros((h, w, 3))
Zimg = np.zeros((h, w, 1))
Rimg = np.zeros((h, w, 1))
Gimg = np.zeros((h, w, 1))
Bimg = np.zeros((h, w, 1))

dmax = 0
for i in range(h):
    print(100*i//h,"%")
    for j in range(w):
        x = Screen[0]*i/h
        y = Screen[1]*j/w
        ray = normalize(np.array([x-Camera[0], y-Camera[1], 0-Camera[2]]))

        col = np.array([0.,0.,0.])
        depth = 0
        reflection = 1.
        O = Camera
        trans = 0
        d = np.array([0.,0.,0.])

        while depth < depth_max:
            tr = trace(O,ray)
            if not tr:
                break
            obj, M, N, col_ray, trans_r = tr
            if depth == 0:
                d = np.linalg.norm(M-O)
                dmax = max(dmax, d)
            depth += 1
            col += reflection * col_ray * (1-trans)
            trans = trans_r
            ray = normalize(ray - 2 * np.dot(ray, N) * N)
            O = M + N * .0001
            reflection *= obj.get('reflection', 1.)
            if reflection == 0.:
                break

        #col += trace(ray)
        #print(col)
        img[h-i-1, j, :] =  np.clip(col, 0, 1)
        Zimg[h-i-1, j, :] =  d
        Rimg[h-i-1, j, :] =  img[h-i-1, j, 0]
        Gimg[h-i-1, j, :] =  img[h-i-1, j, 1]
        Bimg[h-i-1, j, :] =  img[h-i-1, j, 2]
        
        #img[i, j, :] =  np.clip(col, 0, 1)



#Zimg = Zimg / dmax
imgAliasing =  fxaa.onePass(img, Zimg, Rimg, Gimg, Bimg, h, w)



#for i in range(10):
#    for j in range(20):
#        img[h-i-1, j, :] = [1,0,0]

#for i in range(h):
#    for j in range(w):
#        img[h-i-1, j, :] = [0,0,0]

print(Zimg)


plt.imsave('fig.png', img)
plt.imsave('figA.png', imgAliasing)
plt.imshow(img)
plt.show()
#plt.show()
plt.imshow(imgAliasing)
plt.show()