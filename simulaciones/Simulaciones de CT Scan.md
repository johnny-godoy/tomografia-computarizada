# Tarea 1

En este Jupyter Notebook, incluimos una simulación de reconstrucción de imágenes dada su transformada de Radon.

* Curso: Problemas Inversos y de Control de EDP (MA5306)
* Profesor de Cátedra: Axel Osses A.
* Profesor Auxiliar: Jorge Aguayo
* Estudiante: Johnny Godoy

# Setup

Imports


```python
import platform
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import skimage.transform
import tabulate

from perlin_numpy import generate_perlin_noise_2d
```

Configuraciones


```python
print(f"\n{tabulate.tabulate(platform.uname()._asdict().items())}")

plt.style.use('ggplot')
plt.rc('axes', titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams.update({'lines.markeredgewidth': 1})
plt.rcParams.update({'errorbar.capsize': 2})
```

    
    -------  ----------
    system   Windows
    node     GOEDEL
    release  10
    version  10.0.19044
    machine  AMD64
    -------  ----------
    

Constantes


```python
TAMAÑO_IMAGEN = 16
ANGULOS_DE_RAYOS = np.arange(0, 210, 30)
```

Función de utilidad


```python
def ver_imagen(arreglo_imagen: np.ndarray,
               titulo: str,
               etiqueta_x: str = "$x$",
               etiqueta_y: str = "$y$",
               extent: list = None):
    plt.imshow(arreglo_imagen, origin="lower", extent=extent, aspect='auto')
    plt.colorbar()
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.grid()
```

# Simulando datos de una tomografía

Generaremos un arreglo de densidades a través de ruido de Perlin 2D. Este arreglo será considerado desconocido, y es lo que queremos encontrar.


```python
densidades_simuladas = generate_perlin_noise_2d((TAMAÑO_IMAGEN, TAMAÑO_IMAGEN), (2, 2))
ver_imagen(densidades_simuladas, r"Densidad simulada $\rho(x, y)$")
```


    
![png](output_11_0.png)
    


# Calculando la transformada

Esto se logra con `skimage.transform.radon`. Consideramos este arreglo como el cual realmente podemos medir:


```python
transformada_simulada = skimage.transform.radon(densidades_simuladas,
                                                theta=ANGULOS_DE_RAYOS)
ver_imagen(transformada_simulada, "Pérdida de intensidad de rayos simulados",
           "Ángulo del rayo", "Proyección", extent=[0, 180, 0, 16])
plt.xticks = ANGULOS_DE_RAYOS
```

    C:\Users\David\AppData\Local\Programs\Python\Python310\lib\site-packages\skimage\transform\radon_transform.py:75: UserWarning: Radon transform: image must be zero outside the reconstruction circle
      warn('Radon transform: image must be zero outside the '
    


    
![png](output_13_1.png)
    


# Modelo de optimización

Dado $\tilde{R}$ la transformada de Radon medida, podemos encontrar la densidad resolviendo el siguiente problema de optimización:

$$
\min_{\rho\in \mathbb{R}^{16\times 16}} \Vert R(\rho) - \tilde{R} \Vert 
$$


```python
def costo(densidad_cantidata):
    como_imagen = densidad_cantidata.reshape(TAMAÑO_IMAGEN, TAMAÑO_IMAGEN)
    transformada_candidata = skimage.transform.radon(como_imagen, theta=ANGULOS_DE_RAYOS)
    error = transformada_candidata - transformada_simulada
    return (error*error).mean()
```

Suponemos que si bien no tenemos acceso a las densidades, sí lo tenemos a su promedio.

Así, construimos un punto inicial para la optimización como una imagen constante con valor igual a esta densidad promedio en todo punto.


```python
densidades_iniciales = densidades_simuladas.mean()*np.ones_like(densidades_simuladas).flatten()
solucion = scipy.optimize.minimize(fun=costo, x0=densidades_iniciales)
solucion
```




          fun: 1.67669891479261e-07
     hess_inv: array([[ 7.60282762e+02, -3.75646114e+01, -1.43891741e+01, ...,
             6.05326789e+01,  5.74792500e+00, -4.47663312e+01],
           [-3.75646114e+01,  5.62812810e+01,  8.40379694e-01, ...,
            -9.48234812e+00,  6.94437713e-01,  5.34413005e+00],
           [-1.43891741e+01,  8.40379694e-01,  1.51356578e+01, ...,
            -1.37278836e+01, -8.19517768e+00, -5.75070529e-01],
           ...,
           [ 6.05326789e+01, -9.48234812e+00, -1.37278836e+01, ...,
             3.25952950e+01,  1.03393255e+01, -9.33674785e+00],
           [ 5.74792500e+00,  6.94437713e-01, -8.19517768e+00, ...,
             1.03393255e+01,  9.13561452e+00,  1.16619574e+00],
           [-4.47663312e+01,  5.34413005e+00, -5.75070529e-01, ...,
            -9.33674785e+00,  1.16619574e+00,  1.21852659e+01]])
          jac: array([-1.74568100e-06, -7.16395844e-06,  4.77339888e-06,  8.43442086e-07,
            2.45821716e-06, -2.52557353e-07, -8.48157311e-07,  5.10044261e-06,
            2.76735759e-06,  2.83819128e-06,  2.54065551e-06,  7.84642932e-07,
           -1.23332612e-06, -2.46891078e-07,  1.43658484e-06, -4.63931253e-06,
           -1.86137533e-06, -4.02321892e-06, -3.81694839e-07,  2.68793007e-06,
            2.13256750e-06,  2.96399671e-06,  3.55876776e-06,  8.50633555e-08,
           -6.41647446e-06, -7.61823256e-06, -3.83322508e-06, -4.09127428e-06,
           -2.40966255e-06,  3.84157086e-06,  1.49615584e-06, -6.32078252e-06,
           -4.53544190e-06, -3.70443861e-06,  4.60498056e-07,  3.41520255e-06,
            2.79089118e-06,  3.53306715e-06,  2.91372985e-06, -1.00010183e-06,
            4.83365538e-07, -7.03286354e-06, -6.45374132e-06, -2.52913086e-06,
            1.55218558e-06,  1.45551511e-06, -1.19855694e-06, -8.99484909e-06,
           -1.60118509e-06, -3.47875328e-06,  2.08840740e-06,  9.80279857e-08,
            3.29863096e-06,  5.94804588e-06,  4.33237121e-06,  6.09557567e-07,
           -2.12398707e-06, -2.80249145e-06, -1.85087515e-06,  1.48419590e-06,
           -5.06248430e-07,  2.17725718e-06,  2.27641138e-06, -6.96982366e-06,
            3.00567690e-06, -7.32333646e-06,  2.73900030e-07,  2.51474646e-07,
            3.93824182e-06,  4.15550008e-06,  3.52326401e-07,  1.85057922e-06,
           -3.03938681e-06,  4.46828208e-06, -2.00123969e-06, -1.80659275e-06,
            2.13110068e-06,  5.15632259e-06,  1.20475763e-06, -3.41754261e-06,
            1.23868884e-06, -5.11127556e-06, -1.29661275e-06,  3.74493473e-06,
            3.21876525e-06,  4.64462835e-06, -4.26455422e-07,  3.32971979e-08,
            2.97725693e-06, -8.29908132e-07,  1.67294219e-06,  2.67968340e-06,
            4.50842130e-06,  4.68667612e-06,  2.19553795e-06, -2.95803054e-06,
            1.87488869e-07, -3.43764364e-06,  3.10072714e-07,  3.20129877e-06,
            2.00077067e-06,  2.16704555e-06,  2.15278858e-06, -3.13404558e-06,
            1.55731461e-06, -3.16637200e-07,  2.79085931e-06,  1.14357620e-06,
            3.19037227e-06,  7.31566763e-06,  3.12545184e-06, -7.53302594e-06,
           -1.40027377e-06, -8.22764465e-06,  4.63922980e-08,  1.61947982e-06,
            2.73247545e-06, -2.02191267e-06, -1.39354961e-06, -5.12388221e-07,
           -1.04599347e-06, -1.62953033e-06, -2.68207267e-06,  1.94373855e-06,
            1.84761156e-06,  3.55339924e-06, -2.65887410e-06, -7.40247911e-06,
           -7.00465334e-06, -7.17084367e-06,  2.36294562e-06,  9.41183966e-07,
            1.47366741e-07,  4.49539794e-07, -7.78790847e-07,  2.10114374e-06,
           -2.78320153e-06, -1.89631997e-06, -3.45760491e-06, -1.15363029e-06,
            7.77147280e-07,  3.30440544e-06, -9.00580410e-09, -6.21597674e-06,
           -7.49466185e-07, -6.60600893e-06, -2.62291268e-07,  1.03560842e-06,
           -1.11921921e-06,  4.19378846e-06,  2.00082484e-06,  5.62181000e-07,
           -3.12425732e-06, -4.73859055e-06, -4.18288674e-06, -4.94001625e-07,
            5.52406134e-06,  3.67764440e-06, -1.84168422e-06, -3.25683347e-06,
           -3.29755164e-06, -7.95643544e-06,  5.78070622e-08,  2.65357936e-06,
           -2.23661385e-07, -5.20338663e-07,  2.53314279e-06, -1.96317324e-06,
           -1.88024411e-06, -7.74779489e-06, -1.10088473e-06,  2.18724541e-06,
            4.62856287e-06,  5.55746654e-06, -2.05391907e-06, -3.53185273e-06,
            2.61404431e-06, -1.95771014e-06, -9.17819019e-07, -2.58594145e-06,
            2.25389952e-06,  3.43749116e-06,  1.30903192e-06, -4.89066305e-06,
           -3.47995691e-07,  2.20456498e-07,  1.19142265e-06,  5.68760126e-06,
            4.08935550e-06,  4.09970975e-06,  2.34939963e-06, -2.16859353e-06,
            4.66453876e-06, -3.94062886e-06, -7.44180735e-07, -6.37654763e-07,
            7.20108826e-07, -1.06670893e-06, -1.93225045e-06,  3.65840792e-06,
           -3.75940799e-06,  2.71986655e-06,  5.78201544e-07,  3.11015890e-06,
            4.04173507e-06,  4.80112176e-06,  6.04438396e-06, -2.70362287e-06,
           -6.62830246e-07, -3.21262414e-06, -1.38933734e-06, -2.54439609e-06,
           -1.54606946e-06,  4.09244025e-06, -2.38456869e-06, -2.03355636e-06,
           -1.96822052e-06, -2.45751521e-06, -9.47484148e-07, -9.13465287e-07,
            1.88421883e-06,  9.15060044e-06,  4.48507261e-06, -7.19077726e-06,
           -3.81612043e-06, -4.09129261e-06, -4.34705008e-07,  5.08107696e-06,
            5.68147371e-06,  5.77453665e-06,  2.09119976e-06, -2.26827121e-06,
            2.26189423e-06, -2.44855017e-06,  6.20192075e-09,  1.28755458e-06,
            3.95172218e-06,  2.84973296e-06, -1.88720730e-06, -7.70904112e-06,
           -1.24094610e-06, -3.40278970e-06,  5.60686360e-07,  2.93805246e-06,
            8.37743496e-06,  4.46906440e-06,  7.02402436e-06,  2.90226146e-06,
           -5.80873175e-07,  1.62369044e-06, -2.52621042e-07, -4.63590277e-06,
           -8.23321322e-09,  1.59506517e-06,  1.08485360e-06, -5.70035330e-06])
      message: 'Optimization terminated successfully.'
         nfev: 33410
          nit: 129
         njev: 130
       status: 0
      success: True
            x: array([-8.06199618e-03, -1.05685049e-01, -1.98929757e-01, -2.53590830e-01,
           -2.46768587e-01, -1.85360084e-01, -9.90496859e-02, -2.99739011e-02,
            5.72772757e-04,  1.05829957e-03, -6.72932165e-02, -1.85082862e-01,
           -2.91609302e-01, -3.32093568e-01, -2.83306184e-01, -1.60020337e-01,
           -2.67216117e-01, -3.24318405e-01, -3.29551080e-01, -3.75191400e-01,
           -3.90149390e-01, -3.20745011e-01, -2.44809695e-01, -1.70079339e-01,
           -1.86960110e-01, -1.54957798e-01, -2.31315228e-01, -3.56412571e-01,
           -4.00055752e-01, -3.14466048e-01, -2.65118573e-01, -2.30398122e-01,
           -3.80596694e-01, -4.54683297e-01, -4.60447798e-01, -4.69109598e-01,
           -4.50507304e-01, -3.82850248e-01, -2.79584744e-01, -2.28768612e-01,
           -1.77681669e-01, -2.41018396e-01, -3.01796570e-01, -3.85862655e-01,
           -4.51459548e-01, -4.25491544e-01, -3.79321205e-01, -3.43778658e-01,
           -4.38727592e-01, -4.83726445e-01, -4.87079487e-01, -4.44435263e-01,
           -3.58938586e-01, -2.72485559e-01, -1.83861775e-01, -1.71404465e-01,
           -1.18636640e-01, -2.12537876e-01, -2.68583836e-01, -3.58381892e-01,
           -4.22927645e-01, -5.04536219e-01, -4.91532903e-01, -3.90460688e-01,
           -4.28752333e-01, -4.67478934e-01, -3.85534475e-01, -2.77212738e-01,
           -1.36603556e-01, -9.38709689e-02, -1.47501631e-02,  2.96709065e-02,
           -7.58327799e-02, -7.01119288e-02, -1.71517683e-01, -2.56948297e-01,
           -3.07092614e-01, -3.99701354e-01, -4.68116187e-01, -4.07736464e-01,
           -3.72306550e-01, -3.26936563e-01, -1.85711660e-01, -4.65834009e-03,
            6.39424088e-02,  2.23023302e-01,  1.60554171e-01,  1.26166613e-01,
            1.07001622e-01,  1.49369318e-02, -5.00188003e-02, -7.16968554e-02,
           -1.86168266e-01, -2.37229617e-01, -3.50443052e-01, -2.94688694e-01,
           -2.62875816e-01, -1.15128569e-01,  4.56305202e-02,  2.20389604e-01,
            3.35622777e-01,  3.51098866e-01,  3.08910503e-01,  2.76898537e-01,
            1.76674271e-01,  1.16878216e-01,  4.54270300e-02,  1.42090071e-02,
           -2.31769755e-02, -9.05362541e-02, -1.55953300e-01, -2.23069766e-01,
           -1.09620176e-01,  4.29474087e-02,  2.08858588e-01,  3.68172506e-01,
            4.99502697e-01,  3.91956670e-01,  3.69501130e-01,  2.40170488e-01,
            1.18850307e-01,  7.88367154e-02,  6.66580518e-02,  2.32978273e-02,
            5.84742740e-02,  2.71526123e-02,  7.87792418e-03, -7.09472117e-02,
            2.00411699e-02,  1.27513260e-01,  3.41714279e-01,  4.49988272e-01,
            4.11760919e-01,  4.14692221e-01,  2.48742374e-01,  6.30763996e-02,
            2.96679335e-02, -9.44148720e-02, -6.53006632e-02,  9.64911838e-04,
            3.18898952e-02,  9.58927132e-02,  1.41463506e-01,  1.05054893e-01,
            7.50973356e-02,  2.39749631e-01,  3.50533414e-01,  4.03359930e-01,
            4.38486555e-01,  2.46832562e-01,  8.01760067e-02, -7.53170775e-02,
           -1.96949750e-01, -2.40341808e-01, -2.20096459e-01, -1.15018470e-01,
            3.22617048e-02,  1.48915835e-01,  2.66165968e-01,  2.41607884e-01,
            7.91327227e-02,  2.28406102e-01,  2.66412951e-01,  2.83114959e-01,
            2.19413037e-01,  6.66616035e-02, -1.36318438e-01, -2.84982122e-01,
           -4.08355941e-01, -4.37499432e-01, -3.84720857e-01, -2.39921170e-01,
           -3.85946661e-02,  1.43506744e-01,  2.88296893e-01,  2.95288156e-01,
            9.74710068e-03,  4.87341919e-02,  1.05572610e-01,  9.84605677e-02,
           -4.49481217e-02, -1.69357854e-01, -3.61063358e-01, -4.97000307e-01,
           -6.00021415e-01, -5.96835608e-01, -5.19014570e-01, -3.56089586e-01,
           -1.29103606e-01,  1.33298120e-01,  2.35627776e-01,  1.94707359e-01,
           -1.12478254e-01, -7.85421091e-02, -1.07260989e-01, -1.62426618e-01,
           -2.46680251e-01, -3.85608072e-01, -5.17455089e-01, -6.53991345e-01,
           -6.52696010e-01, -6.88525235e-01, -5.81066389e-01, -3.96579454e-01,
           -1.60601126e-01,  7.20136906e-03,  1.16636284e-01,  1.92226847e-01,
           -1.56241885e-01, -2.01617773e-01, -2.83736104e-01, -3.06541736e-01,
           -3.80712739e-01, -4.88736127e-01, -5.55509163e-01, -5.88094656e-01,
           -6.10855769e-01, -5.90770932e-01, -5.20865981e-01, -3.23148831e-01,
           -1.61083368e-01, -2.07991712e-02,  7.45090301e-02, -1.01836596e-02,
           -2.03512139e-01, -2.12279377e-01, -3.57123474e-01, -4.14914869e-01,
           -4.19703439e-01, -4.30016331e-01, -4.29675708e-01, -4.26555046e-01,
           -4.33790173e-01, -3.78142953e-01, -2.92472951e-01, -1.89446370e-01,
           -1.00815450e-01,  4.08531896e-02,  9.71265003e-02, -1.15375191e-01,
           -7.45101688e-02, -1.31612427e-01, -2.13393227e-01, -3.55909967e-01,
           -3.90786574e-01, -3.27370862e-01, -2.89502464e-01, -2.25310354e-01,
           -1.64862189e-01, -1.41129677e-01, -9.17241651e-02,  1.44923436e-02,
            1.43013018e-01,  8.40337481e-02, -3.53004007e-02, -3.76921604e-02])




```python
densidades_estimadas = solucion.x.reshape(TAMAÑO_IMAGEN, TAMAÑO_IMAGEN)
ver_imagen(densidades_estimadas, r"Densidad estimada $\rho(x, y)$")
```


    
![png](output_18_0.png)
    



```python
error = densidades_estimadas - densidades_simuladas
print(f"Error en la densidad: {(error*error).mean()}")
```

    Error en la densidad: 0.0008855905137530617
    

La solución es casi idéntica a lo que buscamos reconstruir, con un error de `1e-07` en las transformadas, y un error de `8e-04` en la imagen original.

Así, se puede reconstruir la imagen (i.e., resolver el problema inverso) con un método simple de optimización, asumiendo que tenemos acceso a una función que calcule la transformada de Radon (i.e., que resuelve el problema directo). 

En la práctica esto no es problema, pues calcular la transformada de Radon solamente requiere integrales a lo largo de un rayo.

Cabe destacar que si bien el supuesto usual es que proveemos de la transformada de Radon completa (es decir, su evaluación para cualquier recta), la reconstrucción solamente requirió utilizar 7 rectas.

Sin embargo, puede ser más desafiente escalar esta metodología para imágenes más grandes, pues una de $16\times16$ ya requirió $20$ segundos para resolver.

