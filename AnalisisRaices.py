import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from control import tf, step_response, impulse_response, bode, nyquist_plot, rlocus, feedback
from control.matlab import ss, ctrb, pole
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp

# Crear ventana principal
root = tk.Tk()
root.title("Linealización de Robot Móvil")
root.geometry("1200x800")

# Crear notebook (pestañas)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=10)

# Pestaña 1: Sistema no lineal
frame_no_lineal = ttk.Frame(notebook)
notebook.add(frame_no_lineal, text="Sistema No Lineal")

# Pestaña 2: Linealización
frame_linealizacion = ttk.Frame(notebook)
notebook.add(frame_linealizacion, text="Proceso de Linealización")

# Pestaña 3: Análisis de Raíces
frame_raices = ttk.Frame(notebook)
notebook.add(frame_raices, text="Análisis de Raíces")

# Pestaña 4: Comparación
frame_comparacion = ttk.Frame(notebook)
notebook.add(frame_comparacion, text="Comparación")

# Variables del sistema
d = 0.5  # Distancia del eje al punto de referencia
x_ref = 0.0
y_ref = 0.0
theta_ref = 0.0

# Sistema no lineal
def sistema_no_lineal(estado, t, v_l, omega):
    x, y, theta = estado
    dxdt = v_l * np.cos(theta) - d * omega * np.sin(theta)
    dydt = v_l * np.sin(theta) + d * omega * np.cos(theta)
    dthetadt = omega
    return [dxdt, dydt, dthetadt]

# Función para linealizar el sistema
def linearizar_sistema(theta_ref, d_val):
    # Definir variables simbólicas
    x, y, theta, v_l, omega = sp.symbols('x y theta v_l omega')
    
    # Ecuaciones no lineales
    f1 = v_l * sp.cos(theta) - d_val * omega * sp.sin(theta)
    f2 = v_l * sp.sin(theta) + d_val * omega * sp.cos(theta)
    f3 = omega
    
    # Calcular Jacobianos
    A = sp.Matrix([[sp.diff(f1, x), sp.diff(f1, y), sp.diff(f1, theta)],
                   [sp.diff(f2, x), sp.diff(f2, y), sp.diff(f2, theta)],
                   [sp.diff(f3, x), sp.diff(f3, y), sp.diff(f3, theta)]])
    
    B = sp.Matrix([[sp.diff(f1, v_l), sp.diff(f1, omega)],
                   [sp.diff(f2, v_l), sp.diff(f2, omega)],
                   [sp.diff(f3, v_l), sp.diff(f3, omega)]])
    
    # Evaluar en el punto de operación
    A_lin = A.subs({x: x_ref, y: y_ref, theta: theta_ref, v_l: 0, omega: 0})
    B_lin = B.subs({x: x_ref, y: y_ref, theta: theta_ref, v_l: 0, omega: 0})
    
    # Convertir a matrices numéricas
    A_np = np.array(A_lin).astype(np.float64)
    B_np = np.array(B_lin).astype(np.float64)
    
    return A_np, B_np

# Función para simular y graficar
def simular_y_mostrar():
    # Obtener parámetros de la interfaz
    try:
        t_final = float(entry_t_final.get())
        v_l_val = float(entry_v_l.get())
        omega_val = float(entry_omega.get())
        x0_val = float(entry_x0.get())
        y0_val = float(entry_y0.get())
        theta0_val = float(entry_theta0.get())
    except:
        print("Error en los parámetros de entrada")
        return
    
    # Tiempo de simulación
    t_sim = np.linspace(0, t_final, 1000)
    
    # Condiciones iniciales
    estado_inicial = [x0_val, y0_val, theta0_val]
    
    # Simular sistema no lineal
    x_no_lineal = odeint(sistema_no_lineal, estado_inicial, t_sim, args=(v_l_val, omega_val))
    
    # Linealizar el sistema
    A_lin, B_lin = linearizar_sistema(theta_ref, d)
    
    # Sistema linealizado
    def sistema_lineal(estado, t):
        x_tilde, y_tilde, theta_tilde = estado
        u_tilde = np.array([v_l_val, omega_val])  # Como v_l_ref=0 y omega_ref=0
        dxdt = A_lin.dot([x_tilde, y_tilde, theta_tilde]) + B_lin.dot(u_tilde)
        return dxdt
    
    # Simular sistema linealizado
    estado_inicial_tilde = [x0_val - x_ref, y0_val - y_ref, theta0_val - theta_ref]
    x_lineal = odeint(sistema_lineal, estado_inicial_tilde, t_sim)
    
    # Convertir a coordenadas absolutas
    x_lineal_abs = x_lineal + np.array([x_ref, y_ref, theta_ref])
    
    # Mostrar resultados en las pestañas
    mostrar_sistema_no_lineal(t_sim, x_no_lineal)
    mostrar_proceso_linealizacion(A_lin, B_lin)
    mostrar_analisis_raices(A_lin, B_lin)
    mostrar_comparacion(t_sim, x_no_lineal, x_lineal_abs)

# Función para mostrar el sistema no lineal
def mostrar_sistema_no_lineal(t_sim, x_no_lineal):
    # Limpiar frame
    for widget in frame_no_lineal.winfo_children():
        widget.destroy()
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfica de trayectoria
    ax1.plot(x_no_lineal[:, 0], x_no_lineal[:, 1], 'b-', linewidth=2)
    ax1.set_xlabel('Posición X')
    ax1.set_ylabel('Posición Y')
    ax1.set_title('Trayectoria del Robot (Sistema No Lineal)')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Gráfica de estados vs tiempo
    ax2.plot(t_sim, x_no_lineal[:, 0], 'r-', label='X')
    ax2.plot(t_sim, x_no_lineal[:, 1], 'g-', label='Y')
    ax2.plot(t_sim, x_no_lineal[:, 2], 'b-', label='θ')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Estados')
    ax2.set_title('Estados vs Tiempo (Sistema No Lineal)')
    ax2.legend()
    ax2.grid(True)
    
    # Mostrar en tkinter
    canvas = FigureCanvasTkAgg(fig, frame_no_lineal)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Función para mostrar el proceso de linealización
def mostrar_proceso_linealizacion(A_lin, B_lin):
    # Limpiar frame
    for widget in frame_linealizacion.winfo_children():
        widget.destroy()
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mostrar matrices
    ax1.axis('off')
    ax1.set_title('Matrices del Sistema Linealizado')
    texto = f"Matriz A:\n{np.array_str(A_lin, precision=3)}\n\nMatriz B:\n{np.array_str(B_lin, precision=3)}"
    ax1.text(0.1, 0.9, texto, transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    
    # Explicación del proceso
    ax2.axis('off')
    ax2.set_title('Proceso de Linealización')
    explicacion = """
    Proceso de Linealización:
    
    1. Se definen las ecuaciones no lineales del sistema:
       dx/dt = v_l·cos(θ) - d·ω·sin(θ)
       dy/dt = v_l·sin(θ) + d·ω·cos(θ)
       dθ/dt = ω
    
    2. Se calculan las matrices Jacobianas A y B:
       A = ∂f/∂x, B = ∂f/∂u
       
    3. Se evalúan las matrices en el punto de operación:
       (x_ref, y_ref, θ_ref, v_l_ref, ω_ref)
       
    4. El sistema linealizado resultante es:
       dẋ = A·(x - x_ref) + B·(u - u_ref)
    """
    ax2.text(0.1, 0.9, explicacion, transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    
    # Mostrar en tkinter
    canvas = FigureCanvasTkAgg(fig, frame_linealizacion)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Función para mostrar el análisis de raíces
def mostrar_analisis_raices(A_lin, B_lin):
    # Limpiar frame
    for widget in frame_raices.winfo_children():
        widget.destroy()
    
    # Crear sistema de espacio de estados
    C = np.eye(3)
    D = np.zeros((3, 2))
    sys = ss(A_lin, B_lin, C, D)
    
    # Obtener polos
    polos = pole(sys)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Diagrama de polos
    ax1.plot(np.real(polos), np.imag(polos), 'rx', markersize=10)
    ax1.axhline(0, color='black', linestyle='--')
    ax1.axvline(0, color='black', linestyle='--')
    ax1.set_xlabel('Parte Real')
    ax1.set_ylabel('Parte Imaginaria')
    ax1.set_title('Diagrama de Polos del Sistema Linealizado')
    ax1.grid(True)
    
    # Análisis de estabilidad
    ax2.axis('off')
    ax2.set_title('Análisis de Estabilidad')
    
    if all(np.real(polos) < 0):
        estabilidad = "Sistema Estable (todos los polos en semiplano izquierdo)"
    elif any(np.real(polos) > 0):
        estabilidad = "Sistema Inestable (al menos un polo en semiplano derecho)"
    else:
        estabilidad = "Sistema Marginalmente Estable (polos en el eje imaginario)"
    
    texto_estabilidad = f"Polos del sistema: {polos}\n\n{estabilidad}"
    
    # Añadir información sobre controllabilidad
    C_mat = ctrb(A_lin, B_lin)
    rango = np.linalg.matrix_rank(C_mat)
    texto_controllabilidad = f"Matriz de Controllabilidad:\nRango = {rango} (de {A_lin.shape[0]})"
    
    if rango == A_lin.shape[0]:
        texto_controllabilidad += "\nSistema completamente controlable"
    else:
        texto_controllabilidad += "\nSistema no completamente controlable"
    
    ax2.text(0.1, 0.9, texto_estabilidad + "\n\n" + texto_controllabilidad, 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    
    # Mostrar en tkinter
    canvas = FigureCanvasTkAgg(fig, frame_raices)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Función para mostrar la comparación
def mostrar_comparacion(t_sim, x_no_lineal, x_lineal_abs):
    # Limpiar frame
    for widget in frame_comparacion.winfo_children():
        widget.destroy()
    
    # Crear figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trayectoria en el plano XY
    ax1.plot(x_no_lineal[:, 0], x_no_lineal[:, 1], 'b-', label='No Lineal', linewidth=2)
    ax1.plot(x_lineal_abs[:, 0], x_lineal_abs[:, 1], 'r--', label='Linealizado', linewidth=2)
    ax1.set_xlabel('Posición X')
    ax1.set_ylabel('Posición Y')
    ax1.set_title('Comparación de Trayectorias')
    ax1.legend()
    ax1.grid(True)
    
    # Posición X
    ax2.plot(t_sim, x_no_lineal[:, 0], 'b-', label='No Lineal', linewidth=2)
    ax2.plot(t_sim, x_lineal_abs[:, 0], 'r--', label='Linealizado', linewidth=2)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Posición X')
    ax2.set_title('Posición X vs Tiempo')
    ax2.legend()
    ax2.grid(True)
    
    # Posición Y
    ax3.plot(t_sim, x_no_lineal[:, 1], 'b-', label='No Lineal', linewidth=2)
    ax3.plot(t_sim, x_lineal_abs[:, 1], 'r--', label='Linealizado', linewidth=2)
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Posición Y')
    ax3.set_title('Posición Y vs Tiempo')
    ax3.legend()
    ax3.grid(True)
    
    # Orientación θ
    ax4.plot(t_sim, x_no_lineal[:, 2], 'b-', label='No Lineal', linewidth=2)
    ax4.plot(t_sim, x_lineal_abs[:, 2], 'r--', label='Linealizado', linewidth=2)
    ax4.set_xlabel('Tiempo (s)')
    ax4.set_ylabel('Orientación θ')
    ax4.set_title('Orientación θ vs Tiempo')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Mostrar en tkinter
    canvas = FigureCanvasTkAgg(fig, frame_comparacion)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Panel de control
frame_control = ttk.Frame(root)
frame_control.pack(fill='x', padx=10, pady=10)

# Etiquetas y campos de entrada
ttk.Label(frame_control, text="Tiempo final:").grid(row=0, column=0, padx=5, pady=5)
entry_t_final = ttk.Entry(frame_control)
entry_t_final.insert(0, "10")
entry_t_final.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_control, text="Velocidad lineal (v_l):").grid(row=0, column=2, padx=5, pady=5)
entry_v_l = ttk.Entry(frame_control)
entry_v_l.insert(0, "0.5")
entry_v_l.grid(row=0, column=3, padx=5, pady=5)

ttk.Label(frame_control, text="Velocidad angular (ω):").grid(row=0, column=4, padx=5, pady=5)
entry_omega = ttk.Entry(frame_control)
entry_omega.insert(0, "0.2")
entry_omega.grid(row=0, column=5, padx=5, pady=5)

ttk.Label(frame_control, text="Posición inicial X:").grid(row=1, column=0, padx=5, pady=5)
entry_x0 = ttk.Entry(frame_control)
entry_x0.insert(0, "0.1")
entry_x0.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame_control, text="Posición inicial Y:").grid(row=1, column=2, padx=5, pady=5)
entry_y0 = ttk.Entry(frame_control)
entry_y0.insert(0, "-0.1")
entry_y0.grid(row=1, column=3, padx=5, pady=5)

ttk.Label(frame_control, text="Orientación inicial θ:").grid(row=1, column=4, padx=5, pady=5)
entry_theta0 = ttk.Entry(frame_control)
entry_theta0.insert(0, "0.05")
entry_theta0.grid(row=1, column=5, padx=5, pady=5)

# Botón para ejecutar la simulación
btn_simular = ttk.Button(frame_control, text="Simular y Linealizar", command=simular_y_mostrar)
btn_simular.grid(row=2, column=0, columnspan=6, pady=10)

# Ejecutar simulación inicial
root.after(100, simular_y_mostrar)

# Ejecutar la aplicación
root.mainloop()
