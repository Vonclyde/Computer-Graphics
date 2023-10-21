Cristian Elías Salas Valera
20.986.693-5

1. Dentro de la escena se observan 5 naves junto a la nave manejable. Esta vez se les añade una estela hecha con triángulos que sigue su movimiento y se torna transparente hacía el final. Para este efecto se usan texturas con bitmap RGBA. Los comandos son los siguientes

W: Hacía adelante
S: Hacía atrás
A: Girar Izquierda
D: Girar Derecha
KEY_UP: Ir hacía arriba
KEY_DOWN: Ir hacía abajo

Las estelas siguen la curva de Bézier de las naves tanto en sentido y dirección. Se pueden hacer estelas más largas/cortas al cambiar la variable 'S' en la función createTrail(). Las estelas se hacen 100% transparentes hacía el final. También son un poco transparentes en el inicio, esto es por la texturas en las cual limpié todos los pixeles de la primera fila, si no hacía esto el final de la estela se veía 100% opaco, arruinando el efecto de transparencia. Por último también tiene un frame "raro" cuando se reinicia la curva, esto es por las naves que vuelven a su ángulo original por un instante, no pude arreglarlo :(.

2. Se añadió una textura realista a todos los planetas y a la luna, más una iluminación con origen en el sol. Se quitó la luz especular a los planetas y se mantiene para el sol, esto para que se viera que el sol está "brillando". Por último a las naves también se les cambia el shader por un lightning shader y así entregar iluminación a las naves, no se les añade textura y se mantuvo una luz especular.

NOTA*: En la carpeta assets se encuentra todo el material utilizado. Las texturas de los planetas son del sitio web Solar System Scope.
