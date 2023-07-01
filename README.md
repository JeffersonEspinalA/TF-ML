<div style="width: 100%; clear: both;">
<div style="float: center; width: 50%;">
<img src="/assets/logo-upc.png", align="center" style="height: 150px; width: 400px;>
</div>

<div style="float: center; width: 50%;">
<h5 style="margin: 0; padding-top: 22px; text-align:center;">Trabajo Final </h5>
<h5 style="margin: 0; padding-top: 22px; text-align:center;">Machine Learning </h5>
<div>
<h5 style="margin: 0; padding-top: 22px; text-align:center;">Integrantes: </h5>

<p>Espinal Atencia, Jefferson William - U201919607</p>
<p>Santisteban Cerna, Jose Mauricio - U201922760</p>
<p>Wu Pan, Tito Peng - U201921200</p>
<p>Caballero Lara, Eduardo Roman - U202019644</p>
</div>


<h5 style="margin: 0; padding-top: 22px; text-align:center;">Profesor: </h5>
<p style="margin: 0; text-align:center;">Canaval Sánchez, Luis Martin</p>
<p style="margin: 0; text-align:center;">Junio 2023</p>
</div>
<div style="width:100%;">&nbsp;</div>

# Indice
1. [Introducción](#data1)
2. [Objetivos](#data2)
3. [Modelo a utilizar ](#data3)
4. [Conjunto de datos utilizados](#data4)
5. [Conversión de OFF a voxels](#data5)
6. [Función de generar huecos al objeto](#data6)
7. [Conversión de voxels a STL](#data7)
8. [Proceso de impresión](#data8)
9. [Objeto Finalizado](#data9)
10. [Bibliografia](#data10)

#### 1. Introducción <a name="data1"></a>
El presente trabajo se centra en el reconocimiento de los formatos de objetos 3D utilizados en el entrenamiento de modelos de machine learning que se especializan en la reconstrucción de objetos con partes faltantes. Tomamos como referente el artículo “3D Reconstruction of Incomplete Archaeological Objects Using a Generative Adversarial Network” de Renato Hermoza e Ivan Sipiran (2018), pues presenta un enfoque innovador que utiliza redes generativas adversariales para lograr una reconstrucción precisa y realista de objetos incompletos. Modelos como el anterior permiten superar los desafíos de la incompletitud en objetos arqueológicos, lo que resulta especialmente relevante para la conservación y estudio de nuestro patrimonio cultural. Para el desarrollo de este trabajo, se abordará la estructura del formato STL, que es ampliamente utilizada en la impresión 3D. El conocimiento de este formato resulta fundamental para la posterior materialización física de los objetos reconstruidos, para lo cual se utilizarán impresoras 3D.


#### 2. Objetivos <a name="data2"></a>
* Identificar los diferentes formatos de objetos tridimensionales utilizados para entrenar un modelo de machine learning.
* Generar modelos para la reconstrucción de objetos con partes faltantes.
* Comprender la estructura del formato STL, el cual se utilizará para imprimir los objetos reconstruidos mediante el uso de impresoras 3D.

#### 3. Modelo a utilizar <a name="data3"></a>
Para este trabajo, el autor del artículo usó un modelo que se basa en el algoritmo GANs que es un tipo de arquitectura de redes neuronales artificiales que consiste en dos componentes principales: el generador y el discriminador.
````python
def _discriminator(dict_size, se=False):
    # inputs
    labels = Input(shape=(1,))
    voxels_inp = Input(shape=(32,32,32))
    voxels = Lambda(lambda x: K.expand_dims(x))(voxels_inp)

    # label embedding
    embs = Embedding(dict_size, 64, input_length=1)(labels)
    embs = Flatten()(embs)
    embs = dense_layer(embs, 1024, act='lrelu', bn=False)

    # conv layers
    out = conv_layer(voxels, 32, 5, 1, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 32, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 64, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 128, act='lrelu', bn=False, se=se)
    out = conv_layer(out, 256, act='lrelu', bn=False, se=se)
    out = Flatten()(out)
    out = dense_layer(out, 1024, act='lrelu', bn=False)
    out = Concatenate()([out, embs])
    out = dense_layer(out, 1024, act='lrelu', bn=False)
    out = dense_layer(out, 512, act='lrelu', bn=False)
    out = dense_layer(out, 1, act=None, bn=False)

    return Model((voxels_inp, labels), out)

def _generator_v(dict_size):
    # inputs
    labels = Input(shape=(1,))
    voxels_inp = Input(shape=(32,32,32))
    voxels = Lambda(lambda x: K.expand_dims(x))(voxels_inp)

    # label embedding
    embs = Embedding(dict_size, 64, input_length=1)(labels)
    embs = Flatten()(embs)
    embs = dense_layer(embs, 1024)

    # conv layers
    out = conv_layer(voxels, 32, 5, 1)
    out = conv_layer(out, 32)
    out = conv_layer(out, 64)
    out = conv_layer(out, 128)
    out = conv_layer(out, 256)

    out = Flatten()(out)
    out = dense_layer(out, 1024)
    out = Concatenate()([out, embs])
    out = dense_layer(out, 1024) # as in D
    out = dense_layer(out, 2*2*2*256)
    out = Lambda(lambda x: K.reshape(x, (-1,2,2,2,256)))(out)

    out = conv_layer(out, 256, transpose=True)
    out = conv_layer(out, 128, transpose=True)
    out = conv_layer(out, 64, transpose=True)
    out = conv_layer(out, 32, transpose=True)

    out = conv_layer(out, 1, 5, 1, act='tanh', bn=False)
    out = Lambda(lambda x: K.squeeze(x, 4))(out)

    return Model((voxels_inp, labels), out)

def _generator_u(dict_size, se=False):
    # inputs
    labels = Input(shape=(1,))
    voxels_inp = Input(shape=(32,32,32))
    voxels = Lambda(lambda x: K.expand_dims(x))(voxels_inp)

    # label embedding
    embs = Embedding(dict_size, 64, input_length=1)(labels)
    embs = Flatten()(embs)
    embs = dense_layer(embs, 1024)

    # conv layers
    encoder1 = conv_layer(voxels, 32, 5, 1, se=se)
    encoder2 = conv_layer(encoder1, 32, se=se)
    encoder3 = conv_layer(encoder2, 64, se=se)
    encoder4 = conv_layer(encoder3, 128, se=se)
    encoder5 = conv_layer(encoder4, 256, se=se)

    mix = Flatten()(encoder5)
    mix = dense_layer(mix, 1024)
    mix = Concatenate()([mix, embs])
    mix = dense_layer(mix, 1024) # as in D
    mix = dense_layer(mix, 2*2*2*256)
    mix = Lambda(lambda x: K.reshape(x, (-1,2,2,2,256)))(mix)
    mix = Concatenate()([mix, encoder5])

    decoder1 = conv_layer(mix, 128, transpose=True, se=se)
    decoder1 = Concatenate()([decoder1, encoder4])
    decoder2 = conv_layer(decoder1, 64, transpose=True, se=se)
    decoder2 = Concatenate()([decoder2, encoder3])
    decoder3 = conv_layer(decoder2, 32, transpose=True, se=se)
    decoder3 = Concatenate()([decoder3, encoder2])
    decoder4 = conv_layer(decoder3, 32, transpose=True, se=se)
    decoder4 = Concatenate()([decoder4, encoder1])

    out = conv_layer(decoder4, 1, 5, 1, act='tanh', bn=False)
    out = Lambda(lambda x: K.squeeze(x, 4))(out)

    return Model((voxels_inp, labels), out)
````
#### 4. Conjunto de datos utilizados <a name="data4"></a>
Los conjuntos de datos utilizados en el artículo se dividen en 2: ModelNet10 y 3D Pottery. El conjunto de datos de datos [ModelNet10](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset), que contiene modelos que están divididos en 10 clases, tales como camas, sillas, escritorios, sofás, mesas, entre otros. Por otro lado, el conjunto de datos 3D Pottery contiene modelos de objetos arqueológicos del museo Larco. En los formatos usados en los conjuntos de datos están disponibles en los formatos OBJ y OFF, estos formatos son normalmente utilizados para representar objetos tridimensionales en aplicaciones de gráficos por computadora y modelado 3D. Finalmente, nuestro decidió escoger el conjunto de datos ModelNet10.
#### 5. Conversión de OFF a voxels <a name="data5"></a>
Para convertir archivos de formato OFF (Object File Format, un formato de archivo 3D) a un formato de voxel (una representación de un objeto 3D en un espacio discreto, similar a un píxel en 2D).
Toma una lista de archivos OFF y convierte cada uno en un array 3D de voxels. Si hay algún problema durante la conversión, se guarda en la lista de errores.
Las funciones necesarias son: 

check_fix_file: lee el archivo OFF y verifica si está correcto el formato
````python
file = random.choice(files)
def check_fix_file(file):
    with open(file) as f:
        l1 = f.readline()
        l2 = f.readlines()

    if l1 != 'OFF\n' and l1[:3] == 'OFF':
        out = 'OFF\n'
        out += l1.split('OFF')[1]
        out += ''.join(l2)
        with open(file, 'w') as f:
            f.write(out)
````
voxels_from_file: llama al ejecutable 'binvox' para convertir el archivo OFF a un archivo voxel (formato binvox)
````python
def voxels_from_file(file, voxsize):
    cmd = f'C:/Users/Jefferson/MachineL/binbox/binvox -d {voxsize} -cb -e {file}'
    check_fix_file(file)
    out_file = file.split('.')[0] + '.binvox'
            
    if os.path.exists(out_file):
        os.remove(out_file)

    t = os.system(cmd)
    
    if t == 0:
        with open(out_file, 'rb') as f:
            d = binvox_rw.read_as_3d_array(f).data
        
        os.remove(out_file)
        return 1, d
    else:
        return 0, None

voxels = voxels_from_file(file, 32)
````
#### 6. Función de generar huecos al objeto <a name="data6"> </a>
Para generar los agujeros aleatorios al objeto se necesitó de la función “get_fractured” que implementó el autor del artículo. Esta función necesita como dato de entrada al objeto en formato voxel. Luego de obtener el resultado de esa función, con la función “plot_vol” se mostrará el objeto fracturado.
La salida del programa será la siguiente:

<img src="/assets/imagen1.png" style="height: 200px; width:200px"/>

#### 7. Conversión de voxels a STL <a name="data7"></a>
Para exportar el objeto roto en formato STL se necesitó convertir el voxel 3d a 2d, que se pudo lograr con la siguiente función:
````python
def convert_to_binary_voxel(input_voxel):
    result = np.zeros((32, 32, 32))
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if(input_voxel[i][j][k] == 1.0):
                    result[i][j][k] = 1
                else:
                    result[i][j][k] = 0
    return result
````
Luego se necesitó de las siguientes lineas de codigo para exportar el voxel bidimensional a STL:
````python
result_voxel_completo = convert_to_binary_voxel(completo)

vertices, faces, _, _ = marching_cubes(result_voxel_completo, level=0)
mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, face in enumerate(faces):
    for j in range(3):
        mesh_data.vectors[i][j] = vertices[face[j], :]

mesh_data.save('resultado_completo.stl')
````
El resultado sería lo siguiente:
<img src="/assets/silla.png" style="height: 200px; width:200px"/>

Además, para las partes faltantes se utilizó la función “logical_xor” de numpy, y lo exportamos a STL de la misma manera:

````python
agujero = np.logical_xor(result_voxel_fracturado, result_voxel_completo)

vertices, faces, _, _ = marching_cubes(agujero, level=0)
mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, face in enumerate(faces):
    for j in range(3):
        mesh_data.vectors[i][j] = vertices[face[j], :]

mesh_data.save('resultado_agujero.stl')
````

#### 8. Proceso de impresión <a name="data8"></a>

##### Objeto mostrado desde el software de impresión

Luego de tener los objetos a imprimir en STL (se convirtió de voxels a STL), el primer paso necesario para la impresión, fue la creación de soportes para una correcta impresión.

<img src="/assets/impresion1.png" style="height: 200px; width:300px"/>

##### Proceso de impresión
<img src="/assets/impresion3.png" style="height: 200px; width:300px"/>
<img src="/assets/impresion2.jpeg" style="height: 200px; width:300px"/>
<img src="/assets/impresion3.jpeg" style="height: 200px; width:300px"/>

#### 9. Objeto finalizado <a name="data8"></a>
<img src="/assets/final1.jpeg" style="height: 200px; width:200px"/>
<img src="/assets/final2.jpeg" style="height: 200px; width:200px"/>

[Video de exposición](link)
#### 10. Bibliografía <a name="data8"></a>
* Ken Douglas (2022) The Main 3D Printing File Formats of 2022.  Consultado de https://all3dp.com/2/3d-file-format-3d-model-types/ [Recuperado el 10 de mayo de 2023]
* Sohail, Shairoz. (2022) “Generating 3D Models with Deep Learning (Part 1)”. Medium. Consultado de https://shairozsohail.medium.com/generating-3d-models-with-deep-learning-part-1-917cc4757143  [Recuperado el 10 de mayo de 2023]
* PyMesh Development Team (2016) PyMesh: Geometry Processing Library for Python. Consultado de https://github.com/PyMesh/PyMesh [Recuperado el 10 de mayo de 2023]
* Dawson-Haggerty, Michael. (2021) Trimesh: Python Library for Loading and Using Triangular Meshes. Consultado de https://github.com/mikedh/trimesh [Recuperado el 10 de mayo de 2023]
