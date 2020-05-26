# Insicoc
Requisitos
- Python 3.7

Antes de instalar los paquetes necesarios, crear un entorno virtual (virtualenv, conda, etc).
Tras activar el entorno virtual, ejecutar **pip install -r requirements.txt**

Con los paquetes instalados, lo siguiente es generar una base de datos con archivos para entrenar y validar la red neuronal
- La estructura es ya está establecida en data
- agregar base de conocimiento (images) respectivamente en su carpeta (perro, gato, gorila)
- si se quiere agregar algún otro animal u objecto, se debe integrar una carpeta adicional con ese objeto, dentro del archivo entrenar cambiar el 3 al número correspondiente.

Tras terminar lo anterior lo siguiente es ejecutar el archivo entrenar
- dependiendo de la maquina será el tiempo de entrenamiento
- personalmente se usó un HP con un procesador AMD A9, el tiempo de espera fue de 16 hrs. Si la maquina tiene una tarjeta gráfica el tiempo será menor.


Al concluir el entrenamiento, se debe crear una carpeta adicional y 2 archivo con el conocimiento de la red.

El archivo Index.py es el encargado de realizar la clasificación, solo agrega el path de la imagen que vamos a comparar y ejecutamos.
