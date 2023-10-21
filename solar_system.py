from cgitb import enable
from pickle import GLOBAL
from pyexpat import model
from re import S, T, X
import glfw
import copy
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
from grafica.transformations import uniformScale
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from grafica.gpu_shape import GPUShape, SIZE_IN_BYTES

## AUTOR: Cristian Elías Salas Valera

class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.X = 0
        self.Y = 0
        self.Z = 0.4
        self.R = 0.25
        self.cameraThetaAngle = np.pi/2 #rotacion con respecto al eje z
        
controller = Controller()

# region Funciones para crear curvas

def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T

def bezierMatrix(P0, P1, P2, P3):
    G = np.concatenate((P0, P1, P2, P3), axis=1)
    Mb = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    return np.matmul(G, Mb)

def evalCurve(M, N):
    ts = np.linspace(0.0, 1.0, N)
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T
        
    return curve

# endregion

# Projeccion y Vista
def setPlot(texture, texturePhongPipeline, simplePhongPipeline, sunPipeline):
    projection = tr.perspective(60, float(width)/float(height), 0.1, 100) #el primer parametro se cambia a 60 para que se vea más escena

    # Fondo de estrellas
    glUseProgram(texture.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texture.shaderProgram, "projection"), 1, GL_TRUE, projection)

    # Planetas
    glUseProgram(texturePhongPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texturePhongPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "Ks"), 0, 0, 0) # Quitamos la reflexión especular a los planetas

    glUniform3f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "lightPosition"), 0, 0, 0)
    glUniform1ui(glGetUniformLocation(texturePhongPipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "constantAttenuation"), 0.5)
    glUniform1f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(texturePhongPipeline.shaderProgram, "quadraticAttenuation"), 0.1)

    #Sol Incandescente
    glUseProgram(sunPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(sunPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0) # Mantenemos la reflexión especular para el sol

    glUniform3f(glGetUniformLocation(sunPipeline.shaderProgram, "lightPosition"), 0, 0, 0)
    glUniform1ui(glGetUniformLocation(sunPipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(sunPipeline.shaderProgram, "constantAttenuation"), 0.5)
    glUniform1f(glGetUniformLocation(sunPipeline.shaderProgram, "linearAttenuation"), 0.5)
    glUniform1f(glGetUniformLocation(sunPipeline.shaderProgram, "quadraticAttenuation"), 0.1)

    # Naves
    glUseProgram(simplePhongPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(simplePhongPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "lightPosition"), 0, 0, 0)
    glUniform1ui(glGetUniformLocation(simplePhongPipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "constantAttenuation"), 0.5)
    glUniform1f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(simplePhongPipeline.shaderProgram, "quadraticAttenuation"), 0.1)

def setView(texture, texturePhongPipeline, simplePhongPipeline, sunPipeline):
    Xesf = controller.R * np.cos(controller.cameraThetaAngle) #coordenada X esferica
    Yesf = controller.R * np.sin(controller.cameraThetaAngle) #coordenada Y esferica

    viewPos = np.array([controller.X-Xesf,controller.Y-Yesf,controller.Z+0.14]) #0.07
    view = tr.lookAt(
            viewPos,
            np.array([controller.X,controller.Y,controller.Z]),
            np.array([0, 0, 1])
        )
    
    glUseProgram(texture.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texture.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(texture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])

    glUseProgram(texturePhongPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texturePhongPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(sunPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(sunPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(simplePhongPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(simplePhongPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

# Funciones para teclas
def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
    else: pass

# Esfera texturizada y normalizada: Sol y planetas
def createLightSphere(N, R):
    vertices = []
    indices = []
    dtheta = (2 * np.pi) / N  #Columnas (Longitudes)
    dphi = np.pi / N          #Filas (Latitudes)

    #Creamos los vértices
    for i in range(N+1):
        phi = (np.pi/2) - i * dphi
        xy = R * np.cos(phi)
        z = R * np.sin(phi)

        for j in range(N+1):
            theta = j * dtheta

            x = xy * np.cos(theta)
            y = xy * np.sin(theta)

            s = float(j/N) 
            t = float(i/N)

            nx = x * float(1/R)
            ny = y * float(1/R)
            nz = z * float(1/R)

                        #pos  #tex      #normals 
            vertices += [x, y, z, s, t, nx, ny, nz]

    #Creamos los índices
    for i in range(N):
        k1 = i * (N + 1)
        k2 = k1 + N + 1

        for m in range(N):
            if (i != 0):
                indices += [k1, k2, k1 + 1]
            if (i != (N-1)):
                indices += [k1 + 1, k2, k2 + 1]
            k1 += 1
            k2 += 1

    return bs.Shape(vertices, indices)

# Esfera texturizada: para fondo de estrellas
def createTexturedSphere(N, R):
    vertices = []
    indices = []
    dtheta = (2 * np.pi) / N  #Columnas (Longitudes)
    dphi = np.pi / N          #Filas (Latitudes)

    #Creamos los vértices
    for i in range(N+1):
        phi = (np.pi/2) - i * dphi
        xy = R * np.cos(phi)
        z = R * np.sin(phi)

        for j in range(N+1):
            theta = j * dtheta

            x = xy * np.cos(theta)
            y = xy * np.sin(theta)

            s = float(j/N) 
            t = float(i/N)
                        #pos  #tex      #normals 
            vertices += [x, y, z, s, t]

    #Creamos los índices
    for i in range(N):
        k1 = i * (N + 1)
        k2 = k1 + N + 1

        for m in range(N):
            if (i != 0):
                indices += [k1, k2, k1 + 1]
            if (i != (N-1)):
                indices += [k1 + 1, k2, k2 + 1]
            k1 += 1
            k2 += 1

    return bs.Shape(vertices, indices)

# Anillo texturizado y normalizado: Usado para el anillo de Saturno
def createRing(N, R, z):
    vertices = []
    indices = []
    dgamma = 2 * np.pi / N
    for m in range(N):
        gamma = m * dgamma

        x = R * np.cos(gamma)
        y = R * np.sin(gamma)
        s=float(m/N)

        # normales
        ux = x * float(1/R)
        uy = y * float(1/R)
        ux_op = R * np.cos(np.pi + gamma) 
        uy_op = R * np.sin(np.pi + gamma)
        uz = 1

        #Añadimos los vértices
        vertices += [x, y, z/8, s, 1/3, ux, uy, uz]
        vertices += [3/5 * x, 3/5 * y, z/8, s, 0, ux_op, uy_op, uz]
        vertices += [3/5 * x, 3/5 * y, -z/8, s, 1, ux_op, uy_op, -uz]
        vertices += [x, y, -z/8, s, 2/3, ux, uy, -uz]
        
    k = 0
    for i in range(N-1):
        indices += [k, k+4, k+7,
                    k+7, k+3, k,
                    k+1, k+5, k+6,
                    k+6, k+2, k+1,
                    k, k+1, k+5,
                    k+5, k+4, k,
                    k+3, k+2, k+6,
                    k+6, k+7, k+3]
        k += 4

    # Indices para juntar los vértices iniciales con los finales
    indices += [k, 0, 3,
                3, k+3, k,
                k+1, 1, 2,
                2, k+2, k+1,
                k, k+1, 1,
                1, 0, k,
                k+3, k+2, 2,
                2, 3, k+3]

    return bs.Shape(vertices, indices)

# Crea Sistema Solar
def createTexturedSolarSystem(pipeline):

    # Facilitamos la inicialización
    def createGPUShape(shape):
        gpuShape = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuShape)
        gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
        return gpuShape

    # Formas Gpu básicas
    gpuMercury = createGPUShape(createLightSphere(24, 1))
    gpuMercury.texture = es.textureSimpleSetup(getAssetPath("mercury.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuVenus = copy.deepcopy(gpuMercury)
    gpuVenus.texture = es.textureSimpleSetup(getAssetPath("venus.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuEarth = copy.deepcopy(gpuMercury)
    gpuEarth.texture = es.textureSimpleSetup(getAssetPath("earth.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuMoon = copy.deepcopy(gpuMercury)
    gpuMoon.texture = es.textureSimpleSetup(getAssetPath("moon.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuMars = copy.deepcopy(gpuMercury)
    gpuMars.texture = es.textureSimpleSetup(getAssetPath("mars.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuJupiter = copy.deepcopy(gpuMercury)
    gpuJupiter.texture = es.textureSimpleSetup(getAssetPath("jupiter.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuSaturn = copy.deepcopy(gpuMercury)
    gpuSaturn.texture = es.textureSimpleSetup(getAssetPath("saturn.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuUranus = copy.deepcopy(gpuMercury)
    gpuUranus.texture = es.textureSimpleSetup(getAssetPath("uranus.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
    gpuNeptune = copy.deepcopy(gpuMercury)
    gpuNeptune.texture = es.textureSimpleSetup(getAssetPath("neptune.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    # Anillos de Saturno
    gpuRing = createGPUShape(createRing(50, 2, 0.4))
    gpuRing.texture = es.textureSimpleSetup(getAssetPath("ring.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    # Leaf Nodes
    mercuryNode = sg.SceneGraphNode("mercuryNode")
    mercuryNode.childs = [gpuMercury]
    venusNode = sg.SceneGraphNode("venusNode")
    venusNode.childs = [gpuVenus]
    marsNode = sg.SceneGraphNode("marsNode")
    marsNode.childs = [gpuMars]
    jupiterNode = sg.SceneGraphNode("jupiterNode")
    jupiterNode.childs = [gpuJupiter]
    uranusNode = sg.SceneGraphNode("uranusNode")
    uranusNode.childs = [gpuUranus]
    neptuneNode = sg.SceneGraphNode("neptuneNode")
    neptuneNode.childs = [gpuNeptune]

    #Tierra y luna
    earthNode = sg.SceneGraphNode("earthNode")
    earthNode.childs = [gpuEarth]
    moonNode = sg.SceneGraphNode("moonNode")
    moonNode.childs = [gpuMoon]

    #Saturno y anillos
    ringNode = sg.SceneGraphNode("outerNode")
    ringNode.childs = [gpuRing]
    saturnNode = sg.SceneGraphNode("saturnNode")
    saturnNode.childs = [gpuSaturn]
    
    # Cuerpos esféricos
    mercury = sg.SceneGraphNode("mercury")
    mercury.childs = [mercuryNode]
    venus = sg.SceneGraphNode("venus")
    venus.childs = [venusNode]
    mars = sg.SceneGraphNode("mars")
    mars.childs = [marsNode]
    jupiter = sg.SceneGraphNode("jupiter")
    jupiter.childs = [jupiterNode]
    uranus = sg.SceneGraphNode("uranus")
    uranus.childs = [uranusNode]
    neptune = sg.SceneGraphNode("neptune")
    neptune.childs = [neptuneNode]

    # Tierra y luna
    earthPlanet = sg.SceneGraphNode("earthPlanet")
    earthPlanet.childs = [earthNode]
    moon = sg.SceneGraphNode("moon")
    moon.childs = [moonNode]
    earth = sg.SceneGraphNode("earth")
    earth.childs = [earthPlanet, moon]

    # Saturno y anillos
    saturnPlanet = sg.SceneGraphNode("saturnPlanet")
    saturnPlanet.childs = [saturnNode]
    Ring = sg.SceneGraphNode("Ring")
    Ring.childs = [ringNode]
    saturn = sg.SceneGraphNode("saturn")
    saturn.childs = [saturnPlanet, Ring]

    #SolarSystem
    solarSystem = sg.SceneGraphNode("solarSystem")
    solarSystem.childs = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]
    return solarSystem

# Crea las naves
def createShipSystem(pipeline):

    def createOFFShip(pipeline, fileOFF, r, g, b):
        ship = bs.readOFF(fileOFF, (r, g, b))
        gpuShip = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuShip)
        gpuShip.fillBuffers(ship.vertices, ship.indices, GL_STATIC_DRAW)
        return gpuShip

    # Naves GPU
    gpuNaboo = createOFFShip(pipeline, getAssetPath('NabooFighter.off'), 1, 1, 0.1)
    gpuDestroyer = createOFFShip(pipeline, getAssetPath('Imperial_star_destroyer.off'), 0.5, 0.5, 0.5)
    gpuKontos = createOFFShip(pipeline, getAssetPath('Kontos.off'), 0.6, 1, 0.6)
    gpuTie = createOFFShip(pipeline, getAssetPath('tie_UV.off'), 1, 0.3, 0)
    gpuFighter= createOFFShip(pipeline, getAssetPath('Tri_Fighter.off'), 0, 1, 1)
    gpuXwing = createOFFShip(pipeline, getAssetPath('XJ5 X-wing starfighter.off'), 1, 0.2, 0.2)

    # Nodos
    nabooNode = sg.SceneGraphNode('nabooNode')
    nabooNode.childs = [gpuNaboo]
    destroyerNode = sg.SceneGraphNode('destroyerNode')
    destroyerNode.childs = [gpuDestroyer]
    kontosNode = sg.SceneGraphNode('kontosNode')
    kontosNode.childs = [gpuKontos]
    tieNode = sg.SceneGraphNode('tieNode')
    tieNode.childs = [gpuTie]
    fighterNode = sg.SceneGraphNode('fighterNode')
    fighterNode.childs = [gpuFighter]
    xwingNode = sg.SceneGraphNode('xwingNode')
    xwingNode.childs = [gpuXwing]
    
    # Naves
    naboo = sg.SceneGraphNode('naboo')
    naboo.childs = [nabooNode]
    destroyer = sg.SceneGraphNode('destroyer')
    destroyer.childs = [destroyerNode]
    kontos = sg.SceneGraphNode('kontos')
    kontos.childs = [kontosNode]
    tie = sg.SceneGraphNode('tie')
    tie.childs = [tieNode]
    fighter = sg.SceneGraphNode('fighter')
    fighter.childs = [fighterNode]
    xwing = sg.SceneGraphNode('xwing')
    xwing.childs = [xwingNode]

    #ShipSystem
    shipSystem = sg.SceneGraphNode('shipsSystem')
    shipSystem.childs = [naboo, destroyer, kontos, tie, fighter, xwing]
    return shipSystem

# Crea las estelas: vértice entrante, vértices acumulados, ángulo, radio, nave, pipeline
def createTrail(vertex, listoftrail, angle, R, ship, pipeline):

    S = 300 # Cantidad de segmentos que conforman la estela

    # Variables globales en la que guardamos los vértices que conforman cada estela
    global trailNaboo
    global trailDestroyer
    global trailKontos
    global trailTie
    global trailFighter
    global trailXwing

    maxTrail = 10 * (S + 1) # Cantidad de vértices totales según la cantidad de segmentos

    def createGPUShape(shape, pipeline):
        gpuShape = es.GPUShape().initBuffers()
        pipeline.setupVAO(gpuShape)
        gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
        return gpuShape

    # Quitamos los 10 últimos elementos de la lista, cada vértice tiene 5 componentes y 
    # debemos quitar el vértice de la izquierda y derecha del último segmento
    if len(trailNaboo) >= maxTrail:
        for x in range(10):
            trailNaboo.pop()
    if len(trailKontos) >= maxTrail:
         for x in range(10):
            trailKontos.pop()
    if len(trailTie) >= maxTrail:
         for x in range(10):
            trailTie.pop()
    if len(trailDestroyer) >= maxTrail:
         for x in range(10):
            trailDestroyer.pop()
    if len(trailFighter) >= maxTrail:
         for x in range(10):
            trailFighter.pop()
    if len(trailXwing) >= maxTrail:
         for x in range(10):
            trailXwing.pop()
    
    # Componentes de cada vértice que entra
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]

    # Añadimos 2 vértices en el diametro del vértice entrante a la función, también se añade la lista que almacena los vértices anteriores dependiendo de la nave
    vertices = [x + R * np.cos(angle), y + R*np.sin(angle), z, 0, 1, x + R * np.cos(np.pi + angle), y + R*np.sin(np.pi + angle), z, 1, 1] + listoftrail
    if ship == 'naboo':
        trailNaboo = vertices
    if ship == 'destroyer':
        trailDestroyer = vertices
    if ship == 'kontos':
        trailKontos = vertices
    if ship == 'tie':
        trailTie = vertices
    if ship == 'fighter':
        trailFighter = vertices
    if ship == 'xwing':
        trailXwing = vertices
    
    # Una vez alcanzada la cantidad máxima de vértices permitidos según los Segmentos S, se obtienen los índices
    if len(vertices) == maxTrail:

        indices = []

        for i in range(S):
            vertices[(10*i)+4] = i/(S)
            vertices[(10*i)+9] = i/(S)

            vertices[(10*S)+4] = 1
            vertices[(10*S)+9] = 1

            k = 2 * i
            indices += [k, k+1, k+2,
                        k+3, k+1, k+2]

        # Se crea la shape y cambiamos la textura según la nave (color de la estela).
        gpuTrail = createGPUShape(bs.Shape(vertices, indices), pipeline)
        if ship == 'naboo':
            gpuTrail.texture = es.textureSimpleSetup(getAssetPath("trail.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        if ship == 'destroyer':
            gpuTrail.texture = es.textureSimpleSetup(getAssetPath("trailDestroyer.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        if ship == 'kontos':
            gpuTrail.texture = es.textureSimpleSetup(getAssetPath("trailKontos.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        if ship == 'tie':
            gpuTrail.texture = es.textureSimpleSetup(getAssetPath("trailTie.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        if ship == 'fighter':
            gpuTrail.texture = es.textureSimpleSetup(getAssetPath("trailFighter.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        if ship == 'xwing':
            gpuTrail.texture = es.textureSimpleSetup(getAssetPath("trailXwing.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)
        
        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())

        # Se dibuja las estelas iterativamente e inmediatamente liberamos el espacio 
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        pipeline.drawCall(gpuTrail)
        glDisable(GL_BLEND)
        gpuTrail.clear()

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    # Creating a glfw window
    width = 800
    height = 800
    title = "Solar System"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)
    glfw.set_key_callback(window, on_key)

    # Creating shader programs
    texturePhongPipeline = ls.SimpleTexturePhongShaderProgram()     # Planetas
    sunPipeline = ls.SimpleTexturePhongShaderProgram()              # Sol
    simplePhongPipeline = ls.SimplePhongShaderProgram()             # Naves
    texture = es.SimpleTextureModelViewProjectionShaderProgram()    # Fondo de estrellas, estelas de naves

    glClearColor(0.5, 0.5, 0.5, 1) # Color de fondo.
    glEnable(GL_DEPTH_TEST) # Habilitamos profundidad al trabajar con cuerpos 3D.
    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # Creamos los planetas
    solarSystem = createTexturedSolarSystem(texturePhongPipeline)
    shipSystem = createShipSystem(simplePhongPipeline)

    # Creamos el sol aparte
    sol = createLightSphere(24, 1)
    gpuSun = es.GPUShape().initBuffers()
    sunPipeline.setupVAO(gpuSun)
    gpuSun.fillBuffers(sol.vertices, sol.indices, GL_STATIC_DRAW)
    gpuSun.texture = es.textureSimpleSetup(getAssetPath("sun.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    # Fondo de estrellas
    StarSphere = createTexturedSphere(24, 100)
    gpuStars = es.GPUShape().initBuffers()
    texture.setupVAO(gpuStars)
    gpuStars.fillBuffers(StarSphere.vertices, StarSphere.indices, GL_STATIC_DRAW)
    gpuStars.texture = es.textureSimpleSetup(getAssetPath("stars.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    sun = sg.SceneGraphNode("sun")
    sun.childs = [gpuSun]
    stars = sg.SceneGraphNode("stars")
    stars.childs = [gpuStars]

    glfw.swap_interval(0)

    # Parametros iniciales
    t0 = glfw.get_time()
    coord_X = 0 
    coord_Y = 0
    coord_Z = 0
    angle = 0
    step = 0

    # Parametros de estela
    trailNaboo = []
    trailDestroyer = []
    trailKontos = []
    trailTie = []
    trailFighter= []
    trailXwing = []

    # region Curvas - Usaremos Curvas de Bézier
    N = 750

    # region Destroyer
    D0 = np.array([[0, -80, 0]]).T
    D1 = np.array([[-120, -80, 0]]).T
    D2 = np.array([[-120, 80, 0]]).T
    D3 = np.array([[0, 80, 0]]).T
    
    M1 = bezierMatrix(D0, D1, D2, D3)
    destroyerCurve1 = evalCurve(M1, N)

    D0 = np.array([[0, 80, 0]]).T
    D1 = np.array([[120, 80, 0]]).T
    D2 = np.array([[120, -80, 0]]).T
    D3 = np.array([[0, -80, 0]]).T
    
    M2 = bezierMatrix(D0, D1, D2, D3)
    destroyerCurve2 = evalCurve(M2, N)

    destroyerC = np.concatenate((destroyerCurve1,destroyerCurve2), axis=0)
    # endregion

    # region Kontos
    K0 = np.array([[0, 80, 0]]).T
    K1 = np.array([[-60, 80, 0]]).T
    K2 = np.array([[60, -80, 0]]).T
    K3 = np.array([[0, -80, 0]]).T
    
    M1 = bezierMatrix(K0, K1, K2, K3)
    kontosCurve1 = evalCurve(M1, N)

    K0 = np.array([[0, -80, 0]]).T
    K1 = np.array([[-60, -80, 0]]).T
    K2 = np.array([[60, 80, 0]]).T
    K3 = np.array([[0, 80, 0]]).T
    
    M2 = bezierMatrix(K0, K1, K2, K3)
    kontosCurve2 = evalCurve(M2, N)

    kontosC = np.concatenate((kontosCurve1,kontosCurve2), axis=0)
    # endregion

    # region Tie
    T0 = np.array([[0, 0, 80]]).T
    T1 = np.array([[140, 0, 80]]).T
    T2 = np.array([[140, 0, -80]]).T
    T3 = np.array([[0, 0, -80]]).T
    
    M1 = bezierMatrix(T0, T1, T2, T3)
    tieCurve1 = evalCurve(M1, N)

    T0 = np.array([[0, 0, -80]]).T
    T1 = np.array([[-140, 0, -80]]).T
    T2 = np.array([[-140, 0, 80]]).T
    T3 = np.array([[0, 0, 80]]).T
    
    M2 = bezierMatrix(T0, T1, T2, T3)
    tieCurve2 = evalCurve(M2, N)

    tieC = np.concatenate((tieCurve1,tieCurve2), axis=0)
    # endregion

    # region Fighter
    F0 = np.array([[0, 80, 0]]).T
    F1 = np.array([[54, 54, 40]]).T
    F2 = np.array([[-54, 28, -40]]).T
    F3 = np.array([[0, 0, 0]]).T
    
    M1 = bezierMatrix(F0, F1, F2, F3)
    fighterCurve1 = evalCurve(M1, N//2)

    F0 = np.array([[0, 0, 0]]).T
    F1 = np.array([[54, -28, 40]]).T
    F2 = np.array([[-54, -54, -40]]).T
    F3 = np.array([[0, -80, 0]]).T
    
    M2 = bezierMatrix(F0, F1, F2, F3)
    fighterCurve2 = evalCurve(M2, N//2)

    fighter_1 = np.concatenate((fighterCurve1, fighterCurve2), axis=0)

    F0 = np.array([[0, -80, 0]]).T
    F1 = np.array([[54, -54, 40]]).T
    F2 = np.array([[-54, -28, -40]]).T
    F3 = np.array([[0, 0, 0]]).T
    
    M3 = bezierMatrix(F0, F1, F2, F3)
    fighterCurve3 = evalCurve(M3, N//2)

    F0 = np.array([[0, 0, 0]]).T
    F1 = np.array([[54, 28, 40]]).T
    F2 = np.array([[-54, 54, -40]]).T
    F3 = np.array([[0, 80, 0]]).T
    
    M4 = bezierMatrix(F0, F1, F2, F3)
    fighterCurve4 = evalCurve(M4, N//2)

    fighter_2 = np.concatenate((fighterCurve3, fighterCurve4), axis=0)
    fighterC = np.concatenate((fighter_1, fighter_2), axis=0)
    # endregion

    # region Xwing
    W0 = np.array([[-100, 0, -100]]).T
    W1 = np.array([[-100, 133, -100]]).T
    W2 = np.array([[100, 133, 100]]).T
    W3 = np.array([[100, 0, 100]]).T
    
    M1 = bezierMatrix(W0, W1, W2, W3)
    xwingCurve1 = evalCurve(M1, N)

    W0 = np.array([[100, 0, 100]]).T
    W1 = np.array([[100, -133, 100]]).T
    W2 = np.array([[-100, -133, -100]]).T
    W3 = np.array([[-100, 0, -100]]).T
    
    M2 = bezierMatrix(W0, W1, W2, W3)
    xwingCurve2 = evalCurve(M2, N)

    xwingC = np.concatenate((xwingCurve1, xwingCurve2), axis=0)
    # endregion

    # endregion

    while not glfw.window_should_close(window):

        perfMonitor.update(glfw.get_time()) #Medimos rendimiento
        glfw.set_window_title(window, title + str(perfMonitor))
        glfw.poll_events() # Chequeamos imputs
 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) #Limpiar pantalla en color y profundidad

        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        # Controles de la nave
        if(glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
            controller.X -= -1 * dt * np.sin(angle)
            controller.Y -= -1 * dt * np.cos(angle) 
            coord_X += 1 * dt * np.sin(angle)  
            coord_Y += 1 * dt * np.cos(angle)  
        if(glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
            controller.X += -1 * dt * np.sin(angle) 
            controller.Y += -1 * dt * np.cos(angle) 
            coord_X -= 1 * dt * np.sin(angle) 
            coord_Y -= 1 * dt * np.cos(angle) 
        if(glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
            controller.cameraThetaAngle += 1.5 * dt 
            angle -= 1.5 * dt 
        if(glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
            controller.cameraThetaAngle -= 1.5 * dt
            angle += 1.5 * dt
        if(glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            controller.Z -= dt
            coord_Z -= dt
        if(glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            controller.Z += dt
            coord_Z += dt

        if (controller.fillPolygon): # Líneas o Polígonos 
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        setPlot(texture, texturePhongPipeline, simplePhongPipeline, sunPipeline)
        setView(texture, texturePhongPipeline, simplePhongPipeline, sunPipeline)

        # region Rotaciones del sol y planetas
        alpha = glfw.get_time()

        mercury = sg.findNode(solarSystem, "mercury")
        mercury.transform = tr.matmul([
            tr.translate(15* np.cos(alpha), 0, 0),          #Rotación elíptica respecto al Sol
            tr.rotationZ(alpha),                            #Rotación respecto al Sol
            tr.translate(24, 0, - 0.8 * np.cos(alpha)),     #Posición respecto al Sol (x) y plano de inclincación (z)
            tr.rotationX(0),                                #Rotación del eje
            tr.rotationZ(0.025*alpha),                      #Rotación sobre su propio eje
            tr.scale(5, 5, 5)
        ])
        venus = sg.findNode(solarSystem, "venus")
        venus.transform = tr.matmul([
            tr.translate(15 * np.cos(0.93 * alpha), 0, 0),
            tr.rotationZ(0.93 * alpha),  
            tr.translate(40, 0, 0.7 * np.cos(alpha)), 
            tr.rotationX(np.pi),
            tr.rotationZ(-0.005 * alpha), 
            tr.scale(5.2, 5.2, 5.2)
        ])
        earthPlanet = sg.findNode(solarSystem, "earthPlanet")
        earthPlanet.transform = tr.matmul([
            tr.rotationX(0.44),
            tr.rotationZ(0.5 * alpha)
        ])
        moon = sg.findNode(solarSystem, "moon")
        moon.transform = tr.matmul([
            tr.rotationZ(alpha), 
            tr.translate(1.5, 0 , 0), 
            tr.rotationX(0.15 + np.pi/2), 
            tr.rotationZ(0.5* alpha), 
            tr.scale(0.18, 0.18, 0.18)
        ])
        earth = sg.findNode(solarSystem, "earth")
        earth.transform = tr.matmul([
            tr.translate(15* np.cos(0.82 * alpha), 0, 0),
            tr.rotationZ(0.82 * alpha), 
            tr.translate(60, 0, 0.6 * np.cos(alpha)), 
            tr.scale(6, 6, 6)
        ])
        mars = sg.findNode(solarSystem, "mars")
        mars.transform = tr.matmul([
            tr.translate(15 * np.cos(0.42 * alpha), 0, 0),
            tr.rotationZ(0.42 * alpha),
            tr.translate(80, 0, -0.67 * np.cos(alpha)),
            tr.rotationX(0.48),
            tr.rotationZ(0.48 * alpha),
            tr.scale(5.4, 5.4, 5.4)
        ])
        jupiter = sg.findNode(solarSystem, "jupiter")
        jupiter.transform = tr.matmul([
            tr.translate(40 * np.cos(0.184 * alpha), 0, 0),
            tr.rotationZ(0.184 * alpha),
            tr.translate(105, 0, 0.843 * np.cos(alpha)),
            tr.rotationX(0.09),
            tr.rotationZ(1.3 * alpha),
            tr.scale(13, 13, 13)
        ])
        saturnPlanet = sg.findNode(solarSystem, "saturnPlanet")
        saturnPlanet.transform = tr.scale(10, 10, 10)

        Ring = sg.findNode(solarSystem, "Ring")
        Ring.transform = tr.matmul([
            tr.scale(10, 10, 10),
            tr.rotationZ(0.85 * alpha)
        ])
        saturn = sg.findNode(solarSystem, "saturn")
        saturn.transform = tr.matmul([
            tr.translate(50 * np.cos(0.113 * alpha), 0, 0),
            tr.rotationZ(0.113 * alpha), 
            tr.translate(145, 0, - np.cos(alpha)),
            tr.rotationX(0.5),
            tr.rotationZ(1.15 * alpha),
        ])
        uranus = sg.findNode(solarSystem, "uranus")
        uranus.transform = tr.matmul([
            tr.translate(40 * np.cos(0.08 * alpha), 0, 0),
            tr.rotationZ(0.08 * alpha),
            tr.translate(205, 0, 0.97 * np.cos(alpha)),
            tr.rotationY(np.pi/2),
            tr.rotationZ(-0.75 * alpha),
            tr.scale(9, 9, 9)
        ])
        neptune = sg.findNode(solarSystem, "neptune")
        neptune.transform = tr.matmul([
            tr.translate(40 * np.cos(0.05 * alpha), 0, 0),
            tr.rotationZ(0.05 * alpha),
            tr.translate(245, 0, -0.757 * np.cos(alpha)),
            tr.rotationY(0.5),
            tr.rotationZ(0.82 * alpha),
            tr.scale(8.8, 8.8, 8.8)
        ])
        # endregion

        # region CURVAS  Y TRANSFORMACIONES
        # Nosotros manejaremos la nave Naboo
        Naboo = sg.findNode(shipSystem, "naboo")
        Naboo.transform = tr.matmul([
            tr.translate(0, 0, 20),
            tr.translate(50*coord_X, 50*coord_Y, 50*coord_Z),
            tr.rotationZ(-angle),
            tr.uniformScale(5),
            tr.rotationZ(np.pi),
            tr.rotationX(np.pi/2)])

        if step > N*2-1:
            step = 0

        if step < N*2-1:
            angle_destroyer = np.arctan2(destroyerC[step+1,0]-destroyerC[step,0], destroyerC[step+1,1]-destroyerC[step,1])
            angle_kontos = np.arctan2(kontosC[step+1,0]-kontosC[step,0], kontosC[step+1,1]-kontosC[step,1])
            angle_tie = np.arctan2(tieC[step+1,0]-tieC[step,0], tieC[step+1,2]-tieC[step,2])
            angle_fighter = np.arctan2(fighterC[step+1,0]-fighterC[step,0], fighterC[step+1,1]-fighterC[step,1])
            angle_xwing = np.arctan2(xwingC[step+1,0]-xwingC[step,0], xwingC[step+1,1]-xwingC[step,1])
        else:
            angle_destroyer = np.arctan2(destroyerC[0,0]-destroyerC[step,0],destroyerC[0,1]-destroyerC[step,1])
            angle_kontos = np.arctan2(kontosC[0,0]-kontosC[step,0],kontosC[0,1]-kontosC[step,1])
            angle_tie = np.arctan2(tieC[0,0]-tieC[step,0], tieC[0,2]-tieC[step,2])
            angle_fighter = np.arctan2(fighterC[0,0]-fighterC[step,0],fighterC[0,1]-fighterC[step,1])
            angle_xwing = np.arctan2(xwingC[0,0]-xwingC[step,0],xwingC[0,1]-xwingC[step,1])

        Destroyer = sg.findNode(shipSystem, "destroyer")
        Destroyer.transform = tr.matmul([
            tr.translate(destroyerC[step,0], destroyerC[step,1], destroyerC[step,2]),
            tr.rotationZ(-angle_destroyer),
            tr.translate(0, 0, 20),
            tr.uniformScale(15),
            tr.rotationZ(np.pi),
            tr.rotationX(np.pi/2)
        ])
        Kontos = sg.findNode(shipSystem, "kontos")
        Kontos.transform = tr.matmul([
            tr.translate(kontosC[step,0], kontosC[step,1], kontosC[step,2]),
            tr.rotationZ(-angle_kontos),
            tr.translate(0, 0, 40),
            tr.uniformScale(10),
            tr.rotationZ(np.pi),
            tr.rotationX(np.pi/2)
        ])
        Tie = sg.findNode(shipSystem, "tie")
        Tie.transform = tr.matmul([
            tr.translate(tieC[step,0], tieC[step,1], tieC[step,2]),
            tr.rotationY(angle_tie),
            tr.rotationZ(-angle_tie),
            tr.uniformScale(15),
            tr.rotationZ(np.pi/2),
            tr.rotationX(np.pi/2)
        ])
        Fighter = sg.findNode(shipSystem, "fighter")
        Fighter.transform = tr.matmul([
            tr.translate(fighterC[step,0], fighterC[step,1], fighterC[step,2]),
            tr.rotationZ(-angle_fighter),
            tr.translate(0, 0, -60),
            tr.uniformScale(15),
            tr.rotationZ(np.pi),
            tr.rotationX(np.pi/2)
        ])
        Xwing = sg.findNode(shipSystem, "xwing")
        Xwing.transform = tr.matmul([
            tr.translate(xwingC[step,0], xwingC[step,1], xwingC[step,2]),
            tr.rotationZ(-angle_xwing),
            tr.uniformScale(10),
            tr.rotationX(np.pi/2)
        ])
        # endregion

        # Dibujamos las escenas - Sistema Solar
        glUseProgram(texturePhongPipeline.shaderProgram)
        sg.drawSceneGraphNode(solarSystem, texturePhongPipeline, "model", 
                np.matmul(tr.translate(0, 0, 0), tr.scale(0.01, 0.01, 0.01)))  

        # Sol
        glUseProgram(sunPipeline.shaderProgram)
        sg.drawSceneGraphNode(sun, sunPipeline, "model", 
        np.matmul(tr.rotationZ(0.1 * alpha), tr.scale(0.15, 0.15, 0.15)))
        
        # Sistema Naves
        glUseProgram(simplePhongPipeline.shaderProgram)
        sg.drawSceneGraphNode(shipSystem, simplePhongPipeline, "model",
                np.matmul(tr.translate(0, 0, 0), tr.scale(0.02, 0.02, 0.02)))
        
        # Fondo de estrellas
        glUseProgram(texture.shaderProgram)
        sg.drawSceneGraphNode(stars, texture, "model", np.matmul(tr.uniformScale(0.5), tr.rotationZ(0.04 * alpha)))
        
        # Estelas para naves autopilotadas
        createTrail([0.02 * destroyerC[step,0], 0.02 * destroyerC[step,1], 0.02 * destroyerC[step,2]+0.42], trailDestroyer, -angle_destroyer, 0.05, 'destroyer', texture)
        createTrail([0.02 * kontosC[step,0], 0.02 * kontosC[step,1], 0.02 * kontosC[step,2] + 0.8], trailKontos, -angle_kontos, 0.02, 'kontos', texture)
        createTrail([0.02 * tieC[step,0], 0.02 * tieC[step,1], 0.02 * tieC[step,2]], trailTie, -angle_tie, 0.02, 'tie', texture)
        createTrail([0.02 * fighterC[step,0], 0.02 * fighterC[step,1], 0.02 * fighterC[step,2]-1.2], trailFighter, -angle_fighter, 0.05, 'fighter', texture)
        createTrail([0.02 * xwingC[step,0], 0.02 * xwingC[step,1], 0.02 * xwingC[step,2]], trailXwing, -angle_xwing, 0.03, 'xwing', texture)

        # Nave controlada
        createTrail([coord_X, coord_Y, coord_Z+0.4], trailNaboo, -angle, 0.005, 'naboo', texture)

        step = step + 1

        glfw.swap_buffers(window)

    solarSystem.clear()
    shipSystem.clear()

    stars.clear()
    sun.clear()

    glfw.terminate()