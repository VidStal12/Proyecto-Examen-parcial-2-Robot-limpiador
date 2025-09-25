# Proyecto Examen parcial 2 , Robot de limpeiza
1. Instalación de WSL (Solo Windows)
1.1. Abrir PowerShell como Administrador
1.2. Ejecutar: `wsl --install`
1.3. Reiniciar el sistema
1.4. Al reiniciar, se completará la instalación automáticamente
Actualizamos el sistema con el siguiente comando en la terminal :sudo apt update && sudo apt upgrade -y 


Video de apoyo para instalación :https://youtu.be/JCpXil0t-Fo?si=RYlSTyCO23PPBQaa


2. Instalar Sagemath con anaconda/miniconda para creación del ambiente

2.1 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


2.2 bash Miniconda3-latest-Linux-x86_64.sh


2.3 Reiniciar terminal después de instalar conda


2.4 (opcional) Sage tiene instalado Python por defecto pero podemos instalarlo de forma independiente con : sudo apt install python3 python3-pip python3-tk -y
tambien instala pip el cual nos permitira instalr librerias en dentro de nuestra distribución de Linux

3.Para Crear un entorno de Sagemath deberemos seguir los siguientes pasos

3.1 conda create -n sageenv sage python=3.11 -c conda-forge -y

3.2 conda activate sageenv


4. Configuración de Visual Studio Code para WSL
   
4.1. Instalar la extensión "Remote - WSL" en VS Code

4.2. Abrir VS Code y presionar Ctrl+Shift+P

4.3. Buscar "WSL: Connect to WSL"

4.4. Seleccionar la distribución Linux instalada (usualmente Ubuntu por defecto al instalar WSL)

Configuración del interprete : 
1. Presionar Ctrl+Shift+P

2. Buscar "Python: Select Interpreter"

3. Seleccionar el intérprete de WSL:

**Si usamos SageMath debe tener esta terminación: ~/miniconda3/envs/sageenv/bin/python

**Si no usamos SageMath: /usr/bin/python3

5. Copiar codigo de archivos a visual estudio con WSL
Copiar codigo a nuevos archivos dentro de wsl por visual estudio ya que es una distribución de Ubuntu y se encunetra separada de nuestro sistema de archivos de windows.

6. Instalacion de librerias

# Si NO usamos SageMath con conda:
pip3 install numpy matplotlib scipy sympy pygame tkinter control

# Si usamos SageMath con conda:
pip install numpy matplotlib scipy pygame control 
# Nota: sympy y tkinter se encuentran incluidos con Sage


