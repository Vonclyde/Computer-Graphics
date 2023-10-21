import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import random
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.transformations as tr
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath

## AUTOR: Cristian Elías Salas Valera

# Parametros iniciales
# Pelota
RADIUS = 0.02
CIRCLE_DISCRETIZATION = 20
CIRCLE_SPEED = 1.2

# Rectángulos
REC_OFFSET = 0.3 # Degradado en el color en la parte superior de los rectángulos
REC_ROWS = 7 # Filas de Rectángulos
MIN_PER_ROW = 7
MAX_PER_ROW = 12

# Player
PLAYER_LENGHT = 3.5
PLAYER_SPEED = 1.4

# Ventana
width = 800
height = 800

class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.win = False
        self.rugosity = False
        
controller = Controller()

# Facilitamos inicialización
def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

# Rectángulo de color
def createRectangle(lenght, width, segments, r, g, b):
    vertices = []
    indices = []

    r0 = r + REC_OFFSET
    g0 = g + REC_OFFSET
    b0 = b + REC_OFFSET
    y= -width

    for i in range(segments+1):
        x = i/segments
        vertices += [lenght * x, 0, 0, r0, g0, b0]
        vertices += [lenght * x, y, 0, r, g, b]

    for i in range(segments):
        k = 2*i
        indices += [k, k+1, k+2]
        indices += [k+2, k+1, k+3]

    return bs.Shape(vertices, indices)

# Se usa los parámetros iniciales dispuestos al inicio del archivo. Esta es una función general para disponer todos los rectángulos, 
def createGame():
    rectangles = []
    l = 1
    for j in range(REC_ROWS):
        REC_NUMBER = random.randint(MIN_PER_ROW, MAX_PER_ROW)
        r, g, b = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)

        k = -1 + (0.1 * 20/(REC_NUMBER+1))/(2 * REC_NUMBER)

        for i in range(REC_NUMBER):
            position = np.array([k, l])
            rectangle = Rectangle(pipeline, position, 20/(REC_NUMBER+1), 4/(0.5 * REC_ROWS), 1, r, g, b)
            rectangle.destroyable = not rectangle.destroyable
            rectangles += [rectangle] 
            k += rectangle.lenght + rectangle.lenght/(REC_NUMBER)
        
        l -= rectangle.width
    return rectangles
        
class Circle:
    def __init__(self, pipeline, position, velocity, r, g, b):
        shape = bs.createColorCircle(CIRCLE_DISCRETIZATION, r, g, b)
        scaleFactor = 2 * RADIUS
        bs.scaleVertices(shape, 6, (scaleFactor, scaleFactor, 1.0))
        self.pipeline = pipeline
        self.gpuShape = createGPUShape(self.pipeline, shape)
        self.position = position
        self.radius = RADIUS
        self.velocity = velocity

    def action(self, gravityAceleration, deltaTime):
        self.velocity += deltaTime * gravityAceleration
        self.position += CIRCLE_SPEED * self.velocity * deltaTime

    def draw(self):
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "transform"), 1, GL_TRUE,
            tr.translate(self.position[0], self.position[1], 0.0)
        )
        self.pipeline.drawCall(self.gpuShape)

class Rectangle:
    def __init__(self, pipeline, position, lenght, width, segments, r, g, b):
        shape = createRectangle(lenght, width, segments, r, g, b)
        bs.scaleVertices(shape, 6, (0.1, 0.1, 1.0))
        self.pipeline = pipeline
        self.gpuShape = createGPUShape(self.pipeline, shape)
        self.position = position
        self.lenght = 0.1 * lenght
        self.width = 0.1 * width
        self.segments = segments
        self.destroyable = False
    
    def draw(self):
        glUniformMatrix4fv(glGetUniformLocation(self.pipeline.shaderProgram, "transform"), 1, GL_TRUE,
            tr.translate(self.position[0], self.position[1], 0.0)
        )
        self.pipeline.drawCall(self.gpuShape)

    def destroy(self):
        self.gpuShape.clear()

# Físicas para las colisiones
# Colision entre círculo y rectángulo
def collide(circle, rectangle):

    assert isinstance(circle, Circle)
    assert isinstance(rectangle, Rectangle)

    # Chocar con un ladrillo por lado izquierdo
    if circle.position[0] < rectangle.position[0]:
        circle.velocity[0] = -abs(circle.velocity[0])

    # Chocar con un ladrillo por lado derecho
    elif circle.position[0] > rectangle.position[0] + rectangle.lenght:
        circle.velocity[0] = abs(circle.velocity[0])

    # Chocar con un ladrillo por abajo
    if circle.position[1] < rectangle.position[1] - rectangle.width:
        circle.velocity[1] = -abs(circle.velocity[1])

    # Chocar con un ladrillo desde arriba
    if circle.position[1] > rectangle.position[1]:

        # Si es que choca con el jugador, cambia según la velocidad[0] según la mitad donde choco
        if rectangle.destroyable == False:
            if circle.position[0] < rectangle.position[0] + rectangle.lenght/2:
                circle.velocity[0] = -abs(circle.velocity[0])
            elif circle.position[0] > rectangle.position[0] + rectangle.lenght/2:
                circle.velocity[0] = abs(circle.velocity[0])
        circle.velocity[1] = abs(circle.velocity[1])

    if rectangle.destroyable: 
        rectangle.destroy()
        
# Se determina si existe colisión entre círculo y rectángulo
def areColliding(circle, rectangle):
    assert isinstance(circle, Circle)
    assert isinstance(rectangle, Rectangle)

    circleDistanceX = abs(circle.position[0]-(rectangle.position[0] + rectangle.lenght/2))
    circleDistanceY = abs(circle.position[1]-(rectangle.position[1] - rectangle.width/2))

    if circleDistanceX > (rectangle.lenght/2 + circle.radius): return False
    if circleDistanceY > (rectangle.width/2 + circle.radius): return False

    if circleDistanceX <= (rectangle.lenght/2): return True
    if circleDistanceY <= (rectangle.width/2): return True

    cornerDistance = (circleDistanceX - rectangle.lenght/2)**2 + (circleDistanceY - rectangle.width/2)**2
    return cornerDistance <= (circle.radius)**2

# Colisión con el borde
def collideWithBorder(circle):

    # Derecha
    if circle.position[0] + circle.radius > 1.0:
        circle.velocity[0] = -abs(circle.velocity[0])
    # Izquierda
    if circle.position[0] < -1.0 + circle.radius:
        circle.velocity[0] = abs(circle.velocity[0])
    # Techo
    if circle.position[1] > 1.0 - circle.radius:
        circle.velocity[1] = -abs(circle.velocity[1])

        # Se le agrega rugosidad al techo para variar la dirección de salida
        if controller.rugosity:
            modifier = random.uniform(-0.4, 0.4)
            if circle.velocity[0] >= 0.6:
                circle.velocity[0] = circle.velocity[0] - abs(modifier)
            elif circle.velocity[0] < -0.6:
                circle.velocity[0] = circle.velocity[0] + abs(modifier)
            else:
                circle.velocity[0] = circle.velocity[0] + modifier

    # Suelo - Se crea una disposición nueva de rectángulos, pues "chocar" abajo representa perder el juego
    if circle.position[1] < -1.0 + circle.radius:
        circle.position = np.array([0.0, 0.0])
        circle.velocity = np.array([0.6, -0.9])

        global rectangles

        for rectangle in rectangles: # Liberamos el espacio al destruir todos los rectángulos
            rectangle.destroy()

        rectangles = createGame() # Se crea un juego nuevo

def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    # Se activa/desactiva rugosidad en el techo, para cambiar la trayectoria de la bola
    elif key == glfw.KEY_LEFT_CONTROL:
        controller.rugosity = not controller.rugosity

    elif key == glfw.KEY_R and controller.win == True:
        controller.win = False
        circle.position = np.array([0.0, 0.0])
        circle.velocity = np.array([0.6, -0.9])

        global rectangles

        for rectangle in rectangles: # Liberamos el espacio al destruir todos los rectángulos
            rectangle.destroy()

        rectangles = createGame()

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    else: pass

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    # Creating a glfw window
    title = "Breakout!"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)
    glfw.set_key_callback(window, on_key)

    # Creating our shader program and telling OpenGL to use it
    pipeline = es.SimpleTransformShaderProgram()
    texture = es.SimpleTextureShaderProgram()
    glUseProgram(pipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)
    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)
    glfw.swap_interval(0)

    # Círculo
    position = np.array([0.15, -0.5]) # (1,1) equivale a la parte superior derecha, lo opuesto es (-1,-1)
    velocity = np.array([-0.65, -0.9])
    acceleration = np.array([0.0, 0.0], dtype=np.float32)
    r1, g1, b1 = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
    circle = Circle(pipeline, position, velocity, r1, g1, b1)

    # Player
    center = -0.1 * (PLAYER_LENGHT - PLAYER_LENGHT/2) # El jugador inicia al centro en el eje x
    position2 = np.array([center, -0.75])
    r2, g2, b2 = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
    player = Rectangle(pipeline, position2, PLAYER_LENGHT, 0.3, 4, r2, g2, b2) # Jugador

    # Win
    shape = bs.createTextureQuad(1,1)
    bs.scaleVertices(shape, 5, [2,2,1])
    gpuWin = es.GPUShape().initBuffers()
    texture.setupVAO(gpuWin)
    gpuWin.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    gpuWin.texture = es.textureSimpleSetup(
        getAssetPath("breakout_win.png"), GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR)

    # Juego
    rectangles = createGame()
    
    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Using the time as the theta parameter
        theta = glfw.get_time()
        deltaTime = perfMonitor.getDeltaTime()

        # Clearing the screen
        glClear(GL_COLOR_BUFFER_BIT)

        if(glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS) and (player.position[0] > -1):
            player.position[0] -= PLAYER_SPEED * deltaTime

        if(glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS) and (player.position[0] + player.lenght < 1):
            player.position[0] += PLAYER_SPEED * deltaTime

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Se calculan las físicas
        circle.action(acceleration, deltaTime)
        collideWithBorder(circle)

        if areColliding(circle, player):
            collide(circle, player)

        for rectangle in rectangles:
            if areColliding(circle, rectangle):
                if rectangle in rectangles:
                    rectangles.remove(rectangle)
                    collide(circle, rectangle)

        # Se detecta que no queden ladrillos, se ha ganado el juego
        if rectangles == []:
            glUseProgram(texture.shaderProgram)
            texture.drawCall(gpuWin)
            if controller.win == False:
                circle.velocity = np.array([0.0, 0.0])
                controller.win = True

        # Se dibujan las figuras mientras no haya victoria
        if controller.win == False:
            glUseProgram(pipeline.shaderProgram)
            for rectangle in rectangles:
                rectangle.draw()

            player.draw()
            circle.draw()

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)
    
    for rectangle in rectangles:
        rectangle.gpuShape.clear()

    circle.gpuShape.clear()
    player.gpuShape.clear()
    gpuWin.clear()
    glfw.terminate()