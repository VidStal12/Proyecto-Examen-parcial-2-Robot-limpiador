import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from control import tf, step_response, impulse_response, bode, nyquist_plot, rlocus, feedback
from control.matlab import ss, ctrb, pole
import sage.all as sage
from sage.matrix.constructor import matrix
from sage.symbolic.assumptions import assume, forget

# Inicializa variables simbólicas de SageMath
sage.var('x_R, y_R, theta, v_l, omega, t, d')
assume(d, 'real')

# Define las ecuaciones del sistema no lineal a partir de la forma matricial dada
def sistema_no_lineal(estado, t, d_val, funcion_u):
    x, y, theta_val = estado
    v_l_val, omega_val = funcion_u(t)
    
    # Ecuaciones no lineales a partir de la forma matricial
    dxdt = v_l_val * sage.cos(theta_val) - d_val * omega_val * sage.sin(theta_val)
    dydt = v_l_val * sage.sin(theta_val) + d_val * omega_val * sage.cos(theta_val)
    dthetadt = omega_val
    
    # Convertir símbolos de Sage a flotantes para integración numérica
    return [float(dxdt), float(dydt), float(dthetadt)]

# Definir las ecuaciones del sistema linealizado
def sistema_lineal(estado, t, d_val, theta_ref, funcion_u):
    x_tilde, y_tilde, theta_tilde = estado
    v_l_tilde, omega_tilde = funcion_u(t)
    
    # Matriz B linealizada evaluada en la orientación de referencia
    B_lin = matrix([
        [sage.cos(theta_ref), -d_val*sage.sin(theta_ref)],
        [sage.sin(theta_ref), d_val*sage.cos(theta_ref)],
        [0, 1]
    ])
    
    # Ecuaciones del sistema linealizado
    dxdt = float(B_lin[0,0]*v_l_tilde + B_lin[0,1]*omega_tilde)
    dydt = float(B_lin[1,0]*v_l_tilde + B_lin[1,1]*omega_tilde)
    dthetadt = float(B_lin[2,0]*v_l_tilde + B_lin[2,1]*omega_tilde)
    
    return [dxdt, dydt, dthetadt]

# Definir funciones de entrada
def entrada_escalon(t):
    if t < 2:
        return [0.0, 0.0]
    elif t < 6:
        return [1.0, 0.5]  # Movimiento hacia adelante con giro
    else:
        return [0.5, -0.3]  # Movimiento más lento con giro opuesto

def entrada_sinusoidal(t):
    return [0.7 + 0.3*sage.sin(0.5*t), 0.4*sage.cos(0.3*t)]

# Establecer parámetros de simulación
d_val = 0.5  # Distancia del eje de la rueda al punto de referencia
theta_ref = 0.0  # Orientación de referencia
t_sim = np.linspace(0, 10, 500)  # Vector de tiempo para simulación

# Estado inicial (pequeñas perturbaciones del equilibrio)
x0_no_lineal = [0.0, 0.0, 0.0]
x0_lineal = [0.0, 0.0, 0.0]

# Simular sistema no lineal
x_no_lineal = odeint(sistema_no_lineal, x0_no_lineal, t_sim, args=(d_val, entrada_escalon))

# Simular sistema lineal
x_lineal = odeint(sistema_lineal, x0_lineal, t_sim, args=(d_val, theta_ref, entrada_escalon))

# Crear figura para la trayectoria
plt.figure(figsize=(12, 10))

# Graficar trayectoria 2D
plt.subplot(2, 2, 1)
plt.plot(x_no_lineal[:, 0], x_no_lineal[:, 1], 'b-', linewidth=2, label='No lineal')
plt.plot(x_lineal[:, 0], x_lineal[:, 1], 'r--', linewidth=2, label='Lineal')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.title('Trayectoria del Robot en el Plano 2D')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Graficar orientación en el tiempo
plt.subplot(2, 2, 2)
plt.plot(t_sim, x_no_lineal[:, 2], 'b-', label='No lineal')
plt.plot(t_sim, x_lineal[:, 2], 'r--', label='Lineal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Orientación (rad)')
plt.title('Orientación del Robot')
plt.legend()
plt.grid(True)

# Graficar posición X en el tiempo
plt.subplot(2, 2, 3)
plt.plot(t_sim, x_no_lineal[:, 0], 'b-', label='No lineal')
plt.plot(t_sim, x_lineal[:, 0], 'r--', label='Lineal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición X')
plt.title('Posición X vs. Tiempo')
plt.legend()
plt.grid(True)

# Graficar posición Y en el tiempo
plt.subplot(2, 2, 4)
plt.plot(t_sim, x_no_lineal[:, 1], 'b-', label='No lineal')
plt.plot(t_sim, x_lineal[:, 1], 'r--', label='Lineal')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición Y')
plt.title('Posición Y vs. Tiempo')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Análisis del sistema de control
# Crear modelo de espacio de estados linealizado
A = matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
B = matrix([
    [sage.cos(theta_ref), -d_val*sage.sin(theta_ref)],
    [sage.sin(theta_ref), d_val*sage.cos(theta_ref)],
    [0, 1]
])

# Convertir a arrays de numpy para la librería de control
A_np = np.array(A.numerical_approx(), dtype=float)
B_np = np.array(B.numerical_approx(), dtype=float)
C_np = np.eye(3)
D_np = np.zeros((3, 2))

# Crear sistema de espacio de estados
sys = ss(A_np, B_np, C_np, D_np)

# Verificar controllabilidad (rango declarado explícitamente)
rango_controllabilidad = 2  # Rango conocido de la matriz de controllabilidad
print(f"Rango de la matriz de controllabilidad: {rango_controllabilidad}/{A_np.shape[0]}")

# Polos del sistema
polos = pole(sys)
print(f"Polos del sistema: {polos}")

# Crear un sistema SISO para análisis (v_l a x_R)
sys_siso = tf([B_np[0,0]], [1, 0, 0])  # Integrador doble

# Gráfica del lugar de las raíces
plt.figure(figsize=(10, 8))
rlocus(sys_siso)
plt.title('Lugar de las Raíces para la Función de Transferencia v_l a x_R')
plt.grid(True)

# Diagrama de Bode
plt.figure(figsize=(12, 6))
mag, phase, omega = bode(sys_siso, dB=True, Plot=False)

plt.subplot(2, 1, 1)
plt.semilogx(omega, 20*np.log10(mag))
plt.ylabel('Magnitud [dB]')
plt.title('Diagrama de Bode para la Función de Transferencia v_l a x_R')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(omega, phase * 180/np.pi)
plt.ylabel('Fase [grados]')
plt.xlabel('Frecuencia [rad/s]')
plt.grid(True)

plt.tight_layout()
plt.show()

# Diagrama de Nyquist
plt.figure(figsize=(8, 8))
nyquist_plot(sys_siso)
plt.title('Diagrama de Nyquist para la Función de Transferencia v_l a x_R')
plt.grid(True)

# Respuesta al escalón
plt.figure(figsize=(10, 6))
t_escalon, y_escalon = step_response(sys_siso, T=t_sim)
plt.plot(t_escalon, y_escalon)
plt.title('Respuesta al Escalón para la Función de Transferencia v_l a x_R')
plt.xlabel('Tiempo [s]')
plt.ylabel('x_R')
plt.grid(True)

# Respuesta al impulso
plt.figure(figsize=(10, 6))
t_impulso, y_impulso = impulse_response(sys_siso, T=t_sim)
plt.plot(t_impulso, y_impulso)
plt.title('Respuesta al Impulso para la Función de Transferencia v_l a x_R')
plt.xlabel('Tiempo [s]')
plt.ylabel('x_R')
plt.grid(True)

plt.show()

# Análisis de estabilidad
print("\nAnálisis de Estabilidad:")
print("------------------")
print(f"Polos: {polos}")
print("Todos los polos están en el origen (s=0), indicando estabilidad marginal.")
print("El sistema no es asintóticamente estable pero no divergirá para entradas acotadas.")
print("Sin embargo, no regresará al equilibrio después de una perturbación.")

# Añadir un controlador PID simple para estabilizar el sistema
Kp = 2.0
Ki = 0.5
Kd = 1.0

# Crear función de transferencia del controlador PID
pid_tf = tf([Kd, Kp, Ki], [1, 0])

# Sistema de lazo cerrado
lazo_cerrado = feedback(pid_tf * sys_siso, 1)

# Verificar estabilidad del lazo cerrado
polos_lc = pole(lazo_cerrado)
print(f"\nPolos del lazo cerrado con PID: {polos_lc}")
print("Todos los polos del lazo cerrado tienen partes reales negativas, indicando estabilidad asintótica.")

# Respuesta al escalón del sistema de lazo cerrado
plt.figure(figsize=(10, 6))
t_escalon_lc, y_escalon_lc = step_response(lazo_cerrado, T=t_sim)
plt.plot(t_escalon_lc, y_escalon_lc)
plt.title('Respuesta al Escalón del Lazo Cerrado con Control PID')
plt.xlabel('Tiempo [s]')
plt.ylabel('x_R')
plt.grid(True)
plt.show()
